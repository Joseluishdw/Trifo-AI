"""
NeuroZero 3.0 — Micro-Transformer Multi-Cabeza (MHSA) con LayerNorm en NumPy Puro.
Incluye:
- Multi-Head Self-Attention (H = 4 cabezas) con proyecciones Q, K, V y salida W_O.
- Layer Normalization exacta (forward y Jacobiano analítico en backward) tras residuales.
- Retropropagación completa hacia el mensaje actual y hacia el contexto conversacional.
- Global L2 Norm Gradient Clipping + Optimizador Adam desde cero.
- Guardado atómico de pesos (.tmp + os.replace).
"""

import os
import time
import numpy as np
from typing import List, Dict, Tuple, Optional


def atomic_replace_file(tmp_path: str, target_path: str, retries: int = 6):
    for attempt in range(retries):
        try:
            os.replace(tmp_path, target_path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.015 * (attempt + 1))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(np.clip(x - x_max, -30.0, 30.0))
    return e_x / np.maximum(np.sum(e_x, axis=axis, keepdims=True), 1e-8)


def swish(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = sigmoid(x)
    return x * s, s


def swish_backward(grad_output: np.ndarray, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    return grad_output * (s + x * s * (1.0 - s))


def l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def layer_norm_forward(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> Tuple[np.ndarray, dict]:
    mean = np.mean(x, axis=-1, keepdims=True)
    x_centered = x - mean
    var = np.mean(x_centered ** 2, axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = x_centered * std_inv
    out = gamma * x_hat + beta
    cache = {
        "x_hat": x_hat,
        "std_inv": std_inv,
        "gamma": gamma,
        "feature_dim": x.shape[-1],
    }
    return out, cache


def layer_norm_backward(
    dout: np.ndarray, cache: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_hat = cache["x_hat"]
    std_inv = cache["std_inv"]
    gamma = cache["gamma"]
    n_feat = float(cache["feature_dim"])

    dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    dx_hat = dout * gamma
    sum_dx_hat = np.sum(dx_hat, axis=-1, keepdims=True)
    sum_dx_hat_x = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
    dx = (std_inv / n_feat) * (n_feat * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x)
    return dx, dgamma, dbeta


def clip_gradients_global_l2(
    grads: Dict[str, np.ndarray], max_norm: float = 3.0
) -> Tuple[Dict[str, np.ndarray], float]:
    sq_sum = 0.0
    for g in grads.values():
        sq_sum += float(np.sum(g.astype(np.float64) ** 2))
    global_norm = float(np.sqrt(sq_sum))
    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-8)
        grads = {k: (v * scale).astype(np.float32) for k, v in grads.items()}
    return grads, global_norm


def build_sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class ScratchNeuralNetwork:
    """
    Micro-Transformer v3 en NumPy:
    Tokens -> Positional Encoding -> Multi-Head Self-Attention (H=4, Q,K,V,W_O)
    -> Residual + LayerNorm1 -> Attention Pooling -> Concat Contexto Multi-Turno
    -> Swish Layer -> Tanh + Residual + LayerNorm2 -> Proyección Latente L2.
    """

    def __init__(
        self,
        vocab_size: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden1_dim: int = 128,
        hidden2_dim: int = 96,
        learning_rate: float = 0.005,
        max_grad_norm: float = 3.0,
        max_seq_len: int = 48,
        seed: int = 42,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) debe ser divisible entre num_heads ({num_heads}).")

        self.rng = np.random.default_rng(seed)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.input_dim = embed_dim * 2
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = embed_dim
        self.lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.max_seq_len = max_seq_len
        self.ctx_scale = 0.45

        self.pos_encoding = build_sinusoidal_positional_encoding(max_seq_len, embed_dim)

        self.W_emb = self.rng.normal(0.0, 0.15, size=(vocab_size, embed_dim)).astype(np.float32)
        self.W_emb[0] = 0.0

        scale_d = np.sqrt(1.0 / embed_dim)
        self.W_Q = self.rng.normal(0.0, scale_d, size=(embed_dim, embed_dim)).astype(np.float32)
        self.W_K = self.rng.normal(0.0, scale_d, size=(embed_dim, embed_dim)).astype(np.float32)
        self.W_V = self.rng.normal(0.0, scale_d, size=(embed_dim, embed_dim)).astype(np.float32)
        self.W_O = self.rng.normal(0.0, scale_d, size=(embed_dim, embed_dim)).astype(np.float32)

        self.gamma1 = np.ones((1, embed_dim), dtype=np.float32)
        self.beta1 = np.zeros((1, embed_dim), dtype=np.float32)
        self.w_pool = self.rng.normal(0.0, scale_d, size=(embed_dim, 1)).astype(np.float32)

        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / self.input_dim), size=(self.input_dim, hidden1_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden1_dim), dtype=np.float32)

        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden1_dim), size=(hidden1_dim, hidden2_dim)).astype(np.float32)
        self.b2 = np.zeros((1, hidden2_dim), dtype=np.float32)
        self.W_res = self.rng.normal(0.0, np.sqrt(1.0 / hidden1_dim), size=(hidden1_dim, hidden2_dim)).astype(np.float32)

        self.gamma2 = np.ones((1, hidden2_dim), dtype=np.float32)
        self.beta2 = np.zeros((1, hidden2_dim), dtype=np.float32)

        self.W3 = self.rng.normal(0.0, np.sqrt(1.0 / hidden2_dim), size=(hidden2_dim, self.output_dim)).astype(np.float32)
        self.b3 = np.zeros((1, self.output_dim), dtype=np.float32)

        self.adam_t = 0
        self.beta1_adam = 0.9
        self.beta2_adam = 0.999
        self.eps = 1e-8
        self._init_adam_moments()

        self.total_epochs_trained = 0
        self.total_steps_trained = 0
        self.last_grad_norm = 0.0
        self.loss_history: List[float] = []
        self.last_token_attention: List[Dict[str, object]] = []
        self.last_head_attention: Dict[str, List[Dict[str, object]]] = {}
        self.last_activations: Dict[str, List[float]] = {
            "input": [0.0] * 16,
            "hidden1": [0.0] * 20,
            "hidden2": [0.0] * 16,
            "output": [0.0] * 16,
        }

    def _get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W_emb": self.W_emb,
            "W_Q": self.W_Q,
            "W_K": self.W_K,
            "W_V": self.W_V,
            "W_O": self.W_O,
            "gamma1": self.gamma1,
            "beta1": self.beta1,
            "w_pool": self.w_pool,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W_res": self.W_res,
            "gamma2": self.gamma2,
            "beta2": self.beta2,
            "W3": self.W3,
            "b3": self.b3,
        }

    def _init_adam_moments(self):
        self.moments: Dict[str, np.ndarray] = {}
        for name, param in self._get_params().items():
            self.moments[f"m_{name}"] = np.zeros_like(param)
            self.moments[f"v_{name}"] = np.zeros_like(param)

    def expand_vocab(self, new_vocab_size: int):
        current_size = self.W_emb.shape[0]
        if new_vocab_size <= current_size:
            return
        diff = new_vocab_size - current_size
        new_rows = self.rng.normal(0.0, 0.15, size=(diff, self.embed_dim)).astype(np.float32)
        self.W_emb = np.vstack([self.W_emb, new_rows])

        zeros_ext = np.zeros((diff, self.embed_dim), dtype=np.float32)
        self.moments["m_W_emb"] = np.vstack([self.moments["m_W_emb"], zeros_ext])
        self.moments["v_W_emb"] = np.vstack([self.moments["v_W_emb"], zeros_ext])

    def update_cooccurrence_embeddings(self, token_ids: List[int], window: int = 3, lr_hebb: float = 0.02):
        valid_ids = [idx for idx in token_ids if 3 < idx < self.W_emb.shape[0]]
        n = len(valid_ids)
        if n < 2:
            return
        for i in range(n):
            u = valid_ids[i]
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if i == j or valid_ids[j] == u:
                    continue
                v = valid_ids[j]
                vec_u = self.W_emb[u].copy()
                vec_v = self.W_emb[v].copy()
                self.W_emb[u] += lr_hebb * (vec_v - vec_u * 0.08)
                norm_u = np.linalg.norm(self.W_emb[u])
                if norm_u > 2.5:
                    self.W_emb[u] = (self.W_emb[u] / norm_u) * 2.5

    def encode_sequence_with_attention(
        self,
        token_ids: List[int],
        tokens_str: Optional[List[str]] = None,
        record_attention: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        if not token_ids:
            zero = np.zeros((1, self.embed_dim), dtype=np.float32)
            return zero, {"empty": True}

        seq_ids = [idx if 0 <= idx < self.W_emb.shape[0] else 1 for idx in token_ids[: self.max_seq_len]]
        L = len(seq_ids)
        H = self.num_heads
        d_h = self.head_dim

        emb = self.W_emb[seq_ids]
        H0 = (emb + 0.22 * self.pos_encoding[:L]).astype(np.float32)

        Q = H0 @ self.W_Q
        K = H0 @ self.W_K
        V = H0 @ self.W_V

        Q_h = Q.reshape(L, H, d_h).transpose(1, 0, 2)
        K_h = K.reshape(L, H, d_h).transpose(1, 0, 2)
        V_h = V.reshape(L, H, d_h).transpose(1, 0, 2)

        scale = 1.0 / np.sqrt(float(d_h))
        scores = np.matmul(Q_h, K_h.transpose(0, 2, 1)) * scale
        attn_weights = softmax(scores, axis=-1)

        context_h = np.matmul(attn_weights, V_h)
        context_concat = context_h.transpose(1, 0, 2).reshape(L, self.embed_dim)
        mhsa_out = context_concat @ self.W_O

        u1 = H0 + mhsa_out
        O, ln1_cache = layer_norm_forward(u1, self.gamma1, self.beta1)

        pool_logits = O @ self.w_pool
        alpha = softmax(pool_logits, axis=0)

        raw_pooled = np.sum(alpha * O, axis=0, keepdims=True)
        raw_norm = np.maximum(np.linalg.norm(raw_pooled, axis=-1, keepdims=True), 1e-6)
        pooled = raw_pooled / raw_norm

        if record_attention and tokens_str:
            words_cut = tokens_str[:L]
            n_w = len(words_cut)
            combined_list = [
                {"token": words_cut[i], "weight": round(float(alpha[i, 0]), 4)}
                for i in range(n_w)
            ]
            head_dict: Dict[str, List[Dict[str, object]]] = {"combined": combined_list}
            for h_idx in range(H):
                mean_col_attn = np.mean(attn_weights[h_idx, :n_w, :n_w], axis=0)
                head_dict[f"head_{h_idx}"] = [
                    {"token": words_cut[i], "weight": round(float(mean_col_attn[i]), 4)}
                    for i in range(n_w)
                ]
            self.last_token_attention = combined_list
            self.last_head_attention = head_dict

        attn_cache = {
            "empty": False,
            "seq_ids": seq_ids,
            "H0": H0,
            "Q_h": Q_h,
            "K_h": K_h,
            "V_h": V_h,
            "attn_weights": attn_weights,
            "context_concat": context_concat,
            "ln1_cache": ln1_cache,
            "O": O,
            "alpha": alpha,
            "raw_norm": raw_norm,
            "pooled": pooled,
        }
        return pooled, attn_cache

    def _backward_sequence_attention(
        self, d_pooled: np.ndarray, attn_cache: dict, acc_grads: Dict[str, np.ndarray]
    ):
        if attn_cache.get("empty", True):
            return

        pooled = attn_cache["pooled"]
        raw_norm = attn_cache["raw_norm"]
        O = attn_cache["O"]
        alpha = attn_cache["alpha"]
        ln1_cache = attn_cache["ln1_cache"]
        context_concat = attn_cache["context_concat"]
        attn_weights = attn_cache["attn_weights"]
        H0 = attn_cache["H0"]
        Q_h = attn_cache["Q_h"]
        K_h = attn_cache["K_h"]
        V_h = attn_cache["V_h"]
        seq_ids = attn_cache["seq_ids"]

        L = len(seq_ids)
        H = self.num_heads
        d_h = self.head_dim
        scale = 1.0 / np.sqrt(float(d_h))

        p_proj = np.sum(d_pooled * pooled, axis=-1, keepdims=True)
        d_raw_pooled = (d_pooled - pooled * p_proj) / raw_norm

        dO = alpha @ d_raw_pooled
        d_alpha = O @ d_raw_pooled.T

        sum_da = np.sum(d_alpha * alpha, axis=0, keepdims=True)
        d_pool_logits = alpha * (d_alpha - sum_da)
        acc_grads["w_pool"] += O.T @ d_pool_logits
        dO = dO + (d_pool_logits @ self.w_pool.T)

        du1, dgamma1, dbeta1 = layer_norm_backward(dO, ln1_cache)
        acc_grads["gamma1"] += dgamma1
        acc_grads["beta1"] += dbeta1

        dH0 = du1.copy()
        d_mhsa_out = du1

        acc_grads["W_O"] += context_concat.T @ d_mhsa_out
        d_context_concat = d_mhsa_out @ self.W_O.T

        d_context_h = d_context_concat.reshape(L, H, d_h).transpose(1, 0, 2)
        dV_h = np.matmul(attn_weights.transpose(0, 2, 1), d_context_h)
        d_attn_weights = np.matmul(d_context_h, V_h.transpose(0, 2, 1))

        sum_dattn = np.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (d_attn_weights - sum_dattn) * scale

        dQ_h = np.matmul(d_scores, K_h)
        dK_h = np.matmul(d_scores.transpose(0, 2, 1), Q_h)

        dQ = dQ_h.transpose(1, 0, 2).reshape(L, self.embed_dim)
        dK = dK_h.transpose(1, 0, 2).reshape(L, self.embed_dim)
        dV = dV_h.transpose(1, 0, 2).reshape(L, self.embed_dim)

        acc_grads["W_Q"] += H0.T @ dQ
        acc_grads["W_K"] += H0.T @ dK
        acc_grads["W_V"] += H0.T @ dV

        dH0 += (dQ @ self.W_Q.T) + (dK @ self.W_K.T) + (dV @ self.W_V.T)

        for pos_i, tok_id in enumerate(seq_ids):
            if 0 < tok_id < self.W_emb.shape[0]:
                acc_grads["W_emb"][tok_id] += dH0[pos_i]

    def encode_context(
        self,
        ctx_token_ids: Optional[List[int]] = None,
        ctx_turns: Optional[List[Tuple[List[int], float]]] = None,
    ) -> Tuple[np.ndarray, dict]:
        valid_turns: List[Tuple[List[int], float]] = []
        if ctx_turns:
            for t_ids, weight in ctx_turns:
                if t_ids and weight > 0.0:
                    valid_turns.append((t_ids, float(weight)))
        elif ctx_token_ids:
            valid_turns.append((ctx_token_ids, 1.0))

        if not valid_turns:
            zero = np.zeros((1, self.embed_dim), dtype=np.float32)
            return zero, {"empty": True}

        total_w = sum(w for _, w in valid_turns)
        norm_turns = [(t_ids, w / max(total_w, 1e-8)) for t_ids, w in valid_turns]

        raw_ctx = np.zeros((1, self.embed_dim), dtype=np.float32)
        turn_caches = []
        for t_ids, w in norm_turns:
            u_vec, t_cache = self.encode_sequence_with_attention(t_ids)
            raw_ctx += w * u_vec
            turn_caches.append((w, t_cache))

        ctx_norm = np.maximum(np.linalg.norm(raw_ctx, axis=-1, keepdims=True), 1e-6)
        ctx_vec = raw_ctx / ctx_norm
        return ctx_vec, {
            "empty": False,
            "ctx_vec": ctx_vec,
            "ctx_norm": ctx_norm,
            "turn_caches": turn_caches,
        }

    def encode_tokens(
        self,
        token_ids: List[int],
        idf_weights: Optional[List[float]] = None,
        tokens_str: Optional[List[str]] = None,
        record_attention: bool = False,
    ) -> np.ndarray:
        pooled, _ = self.encode_sequence_with_attention(
            token_ids, tokens_str=tokens_str, record_attention=record_attention
        )
        return pooled

    def forward(
        self,
        msg_vec: np.ndarray,
        ctx_vec: Optional[np.ndarray] = None,
        record_telemetry: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        if ctx_vec is None:
            ctx_vec = np.zeros_like(msg_vec)

        x = np.hstack([msg_vec, ctx_vec * self.ctx_scale]).astype(np.float32)

        z1 = x @ self.W1 + self.b1
        a1, s1 = swish(z1)

        z2 = a1 @ self.W2 + self.b2
        t2 = np.tanh(z2)
        res = a1 @ self.W_res
        u2 = t2 + 0.35 * res
        a2, ln2_cache = layer_norm_forward(u2, self.gamma2, self.beta2)

        z3 = a2 @ self.W3 + self.b3
        out = l2_normalize(z3)

        cache = {
            "x": x,
            "z1": z1,
            "s1": s1,
            "a1": a1,
            "t2": t2,
            "ln2_cache": ln2_cache,
            "a2": a2,
            "z3": z3,
            "out": out,
        }

        if record_telemetry and x.shape[0] > 0:
            self.last_activations = {
                "input": np.clip(x[0, :16], -2.0, 2.0).tolist(),
                "hidden1": np.clip(a1[0, :20], -2.0, 2.0).tolist(),
                "hidden2": np.clip(a2[0, :16], -2.0, 2.0).tolist(),
                "output": np.clip(out[0, :16], -2.0, 2.0).tolist(),
            }

        return out, cache

    def train_step(
        self,
        msg_token_ids: List[int],
        target_token_ids: List[int],
        ctx_token_ids: Optional[List[int]] = None,
        ctx_turns: Optional[List[Tuple[List[int], float]]] = None,
        negative_token_ids: Optional[List[int]] = None,
        reward_weight: float = 1.0,
    ) -> float:
        msg_vec, msg_attn_cache = self.encode_sequence_with_attention(msg_token_ids)
        ctx_vec, ctx_cache = self.encode_context(ctx_token_ids=ctx_token_ids, ctx_turns=ctx_turns)
        target_vec = self.encode_tokens(target_token_ids)

        _, cache = self.forward(msg_vec, ctx_vec, record_telemetry=True)
        z3 = cache["z3"]
        z3_norm = np.maximum(np.linalg.norm(z3, axis=-1, keepdims=True), 1e-6)
        pred_raw_normed = z3 / z3_norm

        cos_sim = float(np.sum(pred_raw_normed * target_vec))

        if reward_weight >= 0:
            diff = pred_raw_normed - target_vec
            loss = float(0.5 * np.sum(diff ** 2) * reward_weight)
            grad_normed = diff * reward_weight

            if negative_token_ids:
                neg_vec = self.encode_tokens(negative_token_ids)
                neg_sim = float(np.sum(pred_raw_normed * neg_vec))
                margin = 0.22
                if neg_sim > margin:
                    loss += float(0.35 * (neg_sim - margin) ** 2)
                    grad_normed += 0.65 * (neg_sim - margin) * neg_vec
        else:
            punish_strength = abs(reward_weight)
            margin = -0.1
            if cos_sim > margin:
                loss = float(0.5 * ((cos_sim - margin) ** 2) * punish_strength)
                grad_normed = (cos_sim - margin) * target_vec * punish_strength
            else:
                loss = 0.01
                grad_normed = np.zeros_like(pred_raw_normed)

        proj = np.sum(grad_normed * pred_raw_normed, axis=-1, keepdims=True)
        dz3 = (grad_normed - pred_raw_normed * proj) / z3_norm

        a2 = cache["a2"]
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)
        da2 = dz3 @ self.W3.T

        du2, dgamma2, dbeta2 = layer_norm_backward(da2, cache["ln2_cache"])

        t2 = cache["t2"]
        a1 = cache["a1"]
        dz2 = du2 * (1.0 - t2 ** 2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dW_res = a1.T @ (du2 * 0.35)
        da1 = (dz2 @ self.W2.T) + (du2 * 0.35) @ self.W_res.T

        z1 = cache["z1"]
        s1 = cache["s1"]
        x = cache["x"]
        dz1 = swish_backward(da1, z1, s1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        dx = dz1 @ self.W1.T

        grads: Dict[str, np.ndarray] = {
            name: np.zeros_like(param) for name, param in self._get_params().items()
        }
        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W2"] = dW2
        grads["b2"] = db2
        grads["W_res"] = dW_res
        grads["gamma2"] = dgamma2
        grads["beta2"] = dbeta2
        grads["W3"] = dW3
        grads["b3"] = db3

        # 1. Retropropagación hacia el bloque MHSA del mensaje actual
        d_msg_pooled = dx[:, : self.embed_dim]
        self._backward_sequence_attention(d_msg_pooled, msg_attn_cache, grads)

        # 2. Retropropagación completa hacia el bloque MHSA y embeddings del contexto conversacional
        if not ctx_cache.get("empty", True):
            d_ctx_scaled = dx[:, self.embed_dim :] * self.ctx_scale
            c_vec = ctx_cache["ctx_vec"]
            c_norm = ctx_cache["ctx_norm"]
            c_proj = np.sum(d_ctx_scaled * c_vec, axis=-1, keepdims=True)
            d_raw_ctx = (d_ctx_scaled - c_vec * c_proj) / c_norm

            for turn_w, turn_attn_cache in ctx_cache["turn_caches"]:
                self._backward_sequence_attention(d_raw_ctx * turn_w, turn_attn_cache, grads)

        grads["W_emb"][0] = 0.0
        grads, global_norm = clip_gradients_global_l2(grads, max_norm=self.max_grad_norm)
        self.last_grad_norm = round(global_norm, 5)
        self._apply_adam(grads)
        self.W_emb[0] = 0.0

        self.total_steps_trained += 1
        self.loss_history.append(round(float(loss), 5))
        if len(self.loss_history) > 120:
            self.loss_history = self.loss_history[-120:]

        return float(loss)

    def _apply_adam(self, grads: Dict[str, np.ndarray]):
        self.adam_t += 1
        params = self._get_params()
        bias_corr1 = 1.0 - (self.beta1_adam ** self.adam_t)
        bias_corr2 = 1.0 - (self.beta2_adam ** self.adam_t)

        for name, param in params.items():
            g = grads[name]
            m = self.moments[f"m_{name}"]
            v = self.moments[f"v_{name}"]

            m[:] = self.beta1_adam * m + (1.0 - self.beta1_adam) * g
            v[:] = self.beta2_adam * v + (1.0 - self.beta2_adam) * (g ** 2)

            m_hat = m / bias_corr1
            v_hat = v / bias_corr2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def get_nearest_words(self, word_idx: int, idx2word: Dict[int, str], top_k: int = 6) -> List[Dict[str, float]]:
        if word_idx <= 3 or word_idx >= self.W_emb.shape[0]:
            return []
        target = l2_normalize(self.W_emb[word_idx : word_idx + 1])[0]
        all_normed = l2_normalize(self.W_emb)
        sims = all_normed @ target
        results = []
        for idx in np.argsort(-sims):
            idx = int(idx)
            if idx <= 3 or idx == word_idx:
                continue
            w = idx2word.get(idx, "")
            if len(w) <= 2:
                continue
            results.append({"word": w, "similarity": round(float(sims[idx]), 3)})
            if len(results) >= top_k:
                break
        return results

    def parameter_count(self) -> int:
        return int(sum(p.size for p in self._get_params().values()))

    def save_weights(self, filepath: str):
        tmp_path = f"{filepath}.tmp"
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(tmp_path, "wb") as f:
            np.savez(
                f,
                W_emb=self.W_emb,
                W_Q=self.W_Q,
                W_K=self.W_K,
                W_V=self.W_V,
                W_O=self.W_O,
                gamma1=self.gamma1,
                beta1=self.beta1,
                w_pool=self.w_pool,
                W1=self.W1,
                b1=self.b1,
                W2=self.W2,
                b2=self.b2,
                W_res=self.W_res,
                gamma2=self.gamma2,
                beta2=self.beta2,
                W3=self.W3,
                b3=self.b3,
                meta=np.array(
                    [self.total_epochs_trained, self.total_steps_trained, self.adam_t, self.num_heads],
                    dtype=np.int64,
                ),
                loss_history=np.array(self.loss_history, dtype=np.float32),
            )
            f.flush()
            os.fsync(f.fileno())
        atomic_replace_file(tmp_path, filepath)

    def load_weights(self, filepath: str):
        with np.load(filepath) as data:
            required = ("W_emb", "W_Q", "W_K", "W_V", "W_O", "gamma1", "beta1", "gamma2", "beta2")
            if any(k not in data for k in required) or data["W_emb"].shape[1] != self.embed_dim:
                raise ValueError("Pesos incompatibles con Micro-Transformer MHSA v3; se requiere reinicializar.")

            self.W_emb = data["W_emb"].astype(np.float32)
            self.W_Q = data["W_Q"].astype(np.float32)
            self.W_K = data["W_K"].astype(np.float32)
            self.W_V = data["W_V"].astype(np.float32)
            self.W_O = data["W_O"].astype(np.float32)
            self.gamma1 = data["gamma1"].astype(np.float32)
            self.beta1 = data["beta1"].astype(np.float32)
            self.w_pool = data["w_pool"].astype(np.float32)
            self.W1 = data["W1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.W2 = data["W2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)
            self.W_res = data["W_res"].astype(np.float32)
            self.gamma2 = data["gamma2"].astype(np.float32)
            self.beta2 = data["beta2"].astype(np.float32)
            self.W3 = data["W3"].astype(np.float32)
            self.b3 = data["b3"].astype(np.float32)

            meta = data["meta"]
            self.total_epochs_trained = int(meta[0])
            self.total_steps_trained = int(meta[1])
            self.adam_t = int(meta[2])
            self.loss_history = data["loss_history"].astype(float).tolist()
        self._init_adam_moments()
