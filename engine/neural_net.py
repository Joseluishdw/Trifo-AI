"""
Micro-Transformer Neuronal desde Cero en NumPy Puro (Nivel 2).
Arquitectura:
1. Embeddings Dinámicos (V x D) + Codificación Posicional Sinusoidal (Positional Encoding).
2. Bloque de Auto-Atención Entrenable (Scaled Dot-Product Self-Attention: Q, K, V) con
   conexión residual y Attention-Pooling para saber exactamente a qué palabras presta atención.
3. Red Profunda Post-Atención (Swish + Tanh Residual) hacia el espacio latente semántico.
4. Backpropagation exacto de extremo a extremo (incluyendo gradientes de Q, K, V y Embeddings)
   con Optimizador Adam programado desde cero.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(np.clip(x - x_max, -30.0, 30.0))
    return e_x / np.maximum(np.sum(e_x, axis=axis, keepdims=True), 1e-8)


def swish(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = sigmoid(x)
    return x * s, s


def swish_backward(grad_output: np.ndarray, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    d_swish = s + x * s * (1.0 - s)
    return grad_output * d_swish


def l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def build_sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Codificación posicional sinusoidal clásica de Transformers (Vaswani et al.)."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class ScratchNeuralNetwork:
    """
    Motor Neuronal Micro-Transformer en NumPy Puro:
    Secuencia de Tokens -> Positional Encoding -> Self-Attention (Q, K, V)
    -> Attention Pooling -> Concat Contexto -> Swish + Residual MLP -> Espacio Latente.
    """

    def __init__(
        self,
        vocab_size: int = 4,
        embed_dim: int = 64,
        hidden1_dim: int = 128,
        hidden2_dim: int = 96,
        learning_rate: float = 0.006,
        max_seq_len: int = 40,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.embed_dim = embed_dim
        self.input_dim = embed_dim * 2
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = embed_dim
        self.lr = learning_rate
        self.max_seq_len = max_seq_len

        # Codificación posicional precalculada
        self.pos_encoding = build_sinusoidal_positional_encoding(max_seq_len, embed_dim)

        # 1. Matriz de Embeddings Dinámicos (V, D)
        self.W_emb = self.rng.normal(0.0, 0.15, size=(vocab_size, embed_dim)).astype(np.float32)
        self.W_emb[0] = 0.0

        # 2. Pesos del Bloque Transformer Self-Attention (Q, K, V + Vector de Pooling de Atención)
        scale_d = np.sqrt(1.0 / embed_dim)
        self.W_Q = self.rng.normal(0.0, scale_d, size=(embed_dim, embed_dim)).astype(np.float32)
        self.W_K = self.rng.normal(0.0, scale_d, size=(embed_dim, embed_dim)).astype(np.float32)
        self.W_V = self.rng.normal(0.0, scale_d, size=(embed_dim, embed_dim)).astype(np.float32)
        self.w_pool = self.rng.normal(0.0, scale_d, size=(embed_dim, 1)).astype(np.float32)

        # 3. Capas Profundas Feed-Forward Post-Atención
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / self.input_dim), size=(self.input_dim, hidden1_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden1_dim), dtype=np.float32)

        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden1_dim), size=(hidden1_dim, hidden2_dim)).astype(np.float32)
        self.b2 = np.zeros((1, hidden2_dim), dtype=np.float32)

        self.W_res = self.rng.normal(0.0, np.sqrt(1.0 / hidden1_dim), size=(hidden1_dim, hidden2_dim)).astype(np.float32)

        self.W3 = self.rng.normal(0.0, np.sqrt(1.0 / hidden2_dim), size=(hidden2_dim, self.output_dim)).astype(np.float32)
        self.b3 = np.zeros((1, self.output_dim), dtype=np.float32)

        # 4. Optimizador Adam
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self._init_adam_moments()

        # 5. Telemetría en tiempo real
        self.total_epochs_trained = 0
        self.total_steps_trained = 0
        self.loss_history: List[float] = []
        self.last_token_attention: List[Dict[str, object]] = []
        self.last_activations: Dict[str, List[float]] = {
            "input": [0.0] * 16,
            "hidden1": [0.0] * 20,
            "hidden2": [0.0] * 16,
            "output": [0.0] * 16,
        }

    def _get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W_Q": self.W_Q,
            "W_K": self.W_K,
            "W_V": self.W_V,
            "w_pool": self.w_pool,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W_res": self.W_res,
            "W3": self.W3,
            "b3": self.b3,
        }

    def _init_adam_moments(self):
        self.moments = {}
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

    def update_cooccurrence_embeddings(self, token_ids: List[int], window: int = 3, lr_hebb: float = 0.025):
        valid_ids = [idx for idx in token_ids if 3 < idx < self.W_emb.shape[0]]
        n = len(valid_ids)
        if n < 2:
            return
        for i in range(n):
            u = valid_ids[i]
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if i == j:
                    continue
                v = valid_ids[j]
                if u == v:
                    continue
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
        """
        Pasa una secuencia de tokens por el bloque Self-Attention:
        H0 = Emb(tokens) + 0.25 * PosEncoding
        Q = H0 @ W_Q, K = H0 @ W_K, V = H0 @ W_V
        Attn = softmax((Q @ K.T) / sqrt(D))
        O = Attn @ V + H0  (Conexión Residual)
        alpha = softmax(O @ w_pool)
        pooled = L2Norm(sum(alpha * O))
        """
        if not token_ids:
            zero = np.zeros((1, self.embed_dim), dtype=np.float32)
            return zero, {"empty": True}

        seq_ids = [idx if idx < self.W_emb.shape[0] else 1 for idx in token_ids[: self.max_seq_len]]
        L = len(seq_ids)
        emb = self.W_emb[seq_ids]  # (L, D)
        H0 = emb + 0.22 * self.pos_encoding[:L]  # (L, D)

        Q = H0 @ self.W_Q  # (L, D)
        K = H0 @ self.W_K  # (L, D)
        V = H0 @ self.W_V  # (L, D)

        scale = 1.0 / np.sqrt(float(self.embed_dim))
        scores = (Q @ K.T) * scale  # (L, L)
        attn_matrix = softmax(scores, axis=-1)  # (L, L)

        context_seq = attn_matrix @ V  # (L, D)
        O = context_seq + H0  # Residual connection (L, D)

        pool_logits = O @ self.w_pool  # (L, 1)
        alpha = softmax(pool_logits, axis=0)  # (L, 1)

        raw_pooled = np.sum(alpha * O, axis=0, keepdims=True)  # (1, D)
        raw_norm = np.maximum(np.linalg.norm(raw_pooled, axis=-1, keepdims=True), 1e-6)
        pooled = raw_pooled / raw_norm

        if record_attention and tokens_str:
            words_cut = tokens_str[:L]
            self.last_token_attention = [
                {"token": words_cut[i], "weight": round(float(alpha[i, 0]), 3)}
                for i in range(len(words_cut))
            ]

        attn_cache = {
            "empty": False,
            "seq_ids": seq_ids,
            "H0": H0,
            "Q": Q,
            "K": K,
            "V": V,
            "attn_matrix": attn_matrix,
            "O": O,
            "alpha": alpha,
            "raw_pooled": raw_pooled,
            "raw_norm": raw_norm,
            "pooled": pooled,
        }
        return pooled, attn_cache

    def encode_tokens(
        self,
        token_ids: List[int],
        idf_weights: Optional[List[float]] = None,
        tokens_str: Optional[List[str]] = None,
        record_attention: bool = False,
    ) -> np.ndarray:
        """Codifica una lista de IDs usando el bloque Transformer Self-Attention."""
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

        x = np.hstack([msg_vec, ctx_vec * 0.45]).astype(np.float32)

        z1 = x @ self.W1 + self.b1
        a1, s1 = swish(z1)

        z2 = a1 @ self.W2 + self.b2
        t2 = np.tanh(z2)
        res = a1 @ self.W_res
        a2 = t2 + 0.35 * res

        z3 = a2 @ self.W3 + self.b3
        out = l2_normalize(z3)

        cache = {
            "x": x,
            "z1": z1,
            "s1": s1,
            "a1": a1,
            "z2": z2,
            "t2": t2,
            "a2": a2,
            "z3": z3,
            "out": out,
        }

        if record_telemetry and x.shape[0] > 0:
            self.last_activations = {
                "input": np.clip(x[0, :16], -1.5, 1.5).tolist(),
                "hidden1": np.clip(a1[0, :20], -1.5, 1.5).tolist(),
                "hidden2": np.clip(a2[0, :16], -1.5, 1.5).tolist(),
                "output": np.clip(out[0, :16], -1.5, 1.5).tolist(),
            }

        return out, cache

    def train_step(
        self,
        msg_token_ids: List[int],
        target_token_ids: List[int],
        ctx_token_ids: Optional[List[int]] = None,
        negative_token_ids: Optional[List[int]] = None,
        reward_weight: float = 1.0,
    ) -> float:
        """
        Backpropagation completo a través de:
        1. Capas densas residuales (W3, W2, W_res, W1)
        2. Attention Pooling (w_pool)
        3. Bloque Transformer Self-Attention (W_Q, W_K, W_V)
        4. Matriz de Embeddings Léxicos (W_emb)
        """
        msg_vec, attn_cache = self.encode_sequence_with_attention(msg_token_ids)
        ctx_vec = self.encode_tokens(ctx_token_ids) if ctx_token_ids else np.zeros_like(msg_vec)
        target_vec = self.encode_tokens(target_token_ids)

        pred_out, cache = self.forward(msg_vec, ctx_vec, record_telemetry=True)
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

        # Backprop normalización de salida
        proj = np.sum(grad_normed * pred_raw_normed, axis=-1, keepdims=True)
        dz3 = (grad_normed - pred_raw_normed * proj) / z3_norm

        # Backprop MLP
        a2 = cache["a2"]
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)
        da2 = dz3 @ self.W3.T

        t2 = cache["t2"]
        a1 = cache["a1"]
        dz2 = da2 * (1.0 - t2 ** 2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dW_res = a1.T @ (da2 * 0.35)
        da1 = (dz2 @ self.W2.T) + (da2 * 0.35) @ self.W_res.T

        z1 = cache["z1"]
        s1 = cache["s1"]
        x = cache["x"]
        dz1 = swish_backward(da1, z1, s1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        dx = dz1 @ self.W1.T

        # Backprop hacia el Bloque Self-Attention (Q, K, V, w_pool)
        d_msg_pooled = dx[:, : self.embed_dim]  # (1, D)
        dW_Q = np.zeros_like(self.W_Q)
        dW_K = np.zeros_like(self.W_K)
        dW_V = np.zeros_like(self.W_V)
        dw_pool = np.zeros_like(self.w_pool)

        if not attn_cache.get("empty", True):
            pooled = attn_cache["pooled"]
            raw_norm = attn_cache["raw_norm"]
            O = attn_cache["O"]
            alpha = attn_cache["alpha"]  # (L, 1)
            attn_mat = attn_cache["attn_matrix"]  # (L, L)
            H0 = attn_cache["H0"]
            Q = attn_cache["Q"]
            K = attn_cache["K"]
            V = attn_cache["V"]
            seq_ids = attn_cache["seq_ids"]

            # Jacobiano L2 norm en el pooling
            p_proj = np.sum(d_msg_pooled * pooled, axis=-1, keepdims=True)
            d_raw_pooled = (d_msg_pooled - pooled * p_proj) / raw_norm  # (1, D)

            # raw_pooled = sum(alpha * O, axis=0)
            dO = alpha @ d_raw_pooled  # (L, D)
            d_alpha = O @ d_raw_pooled.T  # (L, 1)

            # alpha = softmax(O @ w_pool, axis=0)
            sum_da_alpha = np.sum(d_alpha * alpha, axis=0, keepdims=True)
            d_pool_logits = alpha * (d_alpha - sum_da_alpha)  # (L, 1)
            dw_pool = O.T @ d_pool_logits  # (D, 1)
            dO += d_pool_logits @ self.w_pool.T  # (L, D)

            # O = attn_mat @ V + H0
            dH0 = dO.copy()
            dV = attn_mat.T @ dO  # (L, D)
            d_attn_mat = dO @ V.T  # (L, L)

            # attn_mat = softmax(scores, axis=-1)
            sum_dattn = np.sum(d_attn_mat * attn_mat, axis=-1, keepdims=True)
            scale = 1.0 / np.sqrt(float(self.embed_dim))
            d_scores = attn_mat * (d_attn_mat - sum_dattn) * scale  # (L, L)

            dQ = d_scores @ K  # (L, D)
            dK = d_scores.T @ Q  # (L, D)

            dW_Q = H0.T @ dQ
            dW_K = H0.T @ dK
            dW_V = H0.T @ dV

            dH0 += (dQ @ self.W_Q.T) + (dK @ self.W_K.T) + (dV @ self.W_V.T)

            # Actualizar embeddings de las palabras de entrada
            for pos_i, tok_id in enumerate(seq_ids):
                if 3 < tok_id < self.W_emb.shape[0]:
                    step_emb = np.clip(dH0[pos_i] * (self.lr * 0.45), -0.05, 0.05)
                    self.W_emb[tok_id] -= step_emb

        grads = {
            "W_Q": np.clip(dW_Q, -2.0, 2.0),
            "W_K": np.clip(dW_K, -2.0, 2.0),
            "W_V": np.clip(dW_V, -2.0, 2.0),
            "w_pool": np.clip(dw_pool, -2.0, 2.0),
            "W1": np.clip(dW1, -2.0, 2.0),
            "b1": np.clip(db1, -2.0, 2.0),
            "W2": np.clip(dW2, -2.0, 2.0),
            "b2": np.clip(db2, -2.0, 2.0),
            "W_res": np.clip(dW_res, -2.0, 2.0),
            "W3": np.clip(dW3, -2.0, 2.0),
            "b3": np.clip(db3, -2.0, 2.0),
        }
        self._apply_adam(grads)

        self.total_steps_trained += 1
        self.loss_history.append(round(float(loss), 5))
        if len(self.loss_history) > 120:
            self.loss_history = self.loss_history[-120:]

        return float(loss)

    def _apply_adam(self, grads: Dict[str, np.ndarray]):
        self.adam_t += 1
        params = self._get_params()
        bias_corr1 = 1.0 - (self.beta1 ** self.adam_t)
        bias_corr2 = 1.0 - (self.beta2 ** self.adam_t)

        for name, param in params.items():
            g = grads[name]
            m = self.moments[f"m_{name}"]
            v = self.moments[f"v_{name}"]

            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)

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
        total = self.W_emb.size
        for p in self._get_params().values():
            total += p.size
        return int(total)

    def save_weights(self, filepath: str):
        np.savez(
            filepath,
            W_emb=self.W_emb,
            W_Q=self.W_Q,
            W_K=self.W_K,
            W_V=self.W_V,
            w_pool=self.w_pool,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W_res=self.W_res,
            W3=self.W3,
            b3=self.b3,
            meta=np.array([self.total_epochs_trained, self.total_steps_trained, self.adam_t], dtype=np.int64),
            loss_history=np.array(self.loss_history, dtype=np.float32),
        )

    def load_weights(self, filepath: str):
        data = np.load(filepath)
        if data["W_emb"].shape[1] != self.embed_dim or "W_Q" not in data:
            raise ValueError("Arquitectura actualizada a Micro-Transformer v2; requiere re-entrenar pesos.")
        self.W_emb = data["W_emb"].astype(np.float32)
        self.W_Q = data["W_Q"].astype(np.float32)
        self.W_K = data["W_K"].astype(np.float32)
        self.W_V = data["W_V"].astype(np.float32)
        self.w_pool = data["w_pool"].astype(np.float32)
        self.W1 = data["W1"].astype(np.float32)
        self.b1 = data["b1"].astype(np.float32)
        self.W2 = data["W2"].astype(np.float32)
        self.b2 = data["b2"].astype(np.float32)
        self.W_res = data["W_res"].astype(np.float32)
        self.W3 = data["W3"].astype(np.float32)
        self.b3 = data["b3"].astype(np.float32)
        meta = data["meta"]
        self.total_epochs_trained = int(meta[0])
        self.total_steps_trained = int(meta[1])
        self.adam_t = int(meta[2])
        self.loss_history = data["loss_history"].astype(float).tolist()
        self._init_adam_moments()
