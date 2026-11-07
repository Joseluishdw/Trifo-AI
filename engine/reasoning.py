"""
NeuroZero 3.0 — Motor de Razonamiento Simbólico, Backward Chaining y Detección de Contradicciones.
Incluye:
- Encadenamiento hacia atrás (Backward Chaining) con árbol de prueba completo para preguntas "¿Por qué X es Y?".
- Encadenamiento hacia adelante (Forward Chaining) multi-salto y Modus Ponens.
- Detector de contradicciones lógicas al insertar nuevos hechos o reglas condicionales.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from .tokenizer import strip_accents, spanish_stem


ISA_RELATIONS = {"es", "es un", "es una", "es el", "es la", "son", "pertenece a"}


def normalize_entity_key(text: str) -> str:
    t = strip_accents(text.lower().strip())
    t = re.sub(r"[^\w\sñ]", " ", t)
    ignore = {"el", "la", "los", "las", "un", "una", "unos", "unas", "todo", "todos", "toda", "todas"}
    words = [w for w in t.split() if w not in ignore]
    if not words:
        return t.strip()
    return " ".join(spanish_stem(w) for w in words)


def adapt_plural_to_singular_verb(rel: str) -> str:
    mapping = {
        "son": "es",
        "no son": "no es",
        "tienen": "tiene",
        "no tienen": "no tiene",
        "pueden": "puede",
        "no pueden": "no puede",
        "necesitan": "necesita",
        "no necesitan": "no necesita",
        "viven en": "vive en",
        "comen": "come",
        "usan": "usa",
        "hacen": "hace",
        "sirven para": "sirve para",
    }
    return mapping.get(rel.lower().strip(), rel)


def parse_relation_polarity(rel: str) -> Tuple[bool, str]:
    """Separa una relación en (es_afirmativa, familia_relacional_canónica)."""
    clean = strip_accents(rel.lower().strip())
    clean = re.sub(r"\s+", " ", clean)
    polarity = True
    if clean.startswith("no "):
        polarity = False
        clean = clean[3:].strip()

    if clean in ISA_RELATIONS:
        return polarity, "isa"
    if clean in {"puede", "pueden"}:
        return polarity, "puede"
    if clean in {"necesita", "necesitan"}:
        return polarity, "necesita"
    if clean in {"tiene", "tienen"}:
        return polarity, "tiene"
    return polarity, adapt_plural_to_singular_verb(clean)


def singularize_word_if_needed(word: str, orig_rel: str) -> str:
    w = word.strip()
    if orig_rel in {"son", "no son"} and " " not in w:
        if w.endswith("es") and len(w) > 4 and w[-3] not in ("a", "e", "i", "o", "u"):
            return w[:-2]
        if w.endswith("s") and len(w) > 3:
            return w[:-1]
    return w


class LogicalReasoningEngine:
    def __init__(self):
        self.rules: List[Dict[str, object]] = []
        self.next_rule_id: int = 1

    def check_rule_contradiction(self, condition: str, consequence: str) -> Optional[Dict[str, object]]:
        cond_key = normalize_entity_key(condition)
        cons_norm = strip_accents(consequence.lower().strip())
        cons_neg = cons_norm.startswith("no ")
        cons_core = normalize_entity_key(cons_norm[3:] if cons_neg else cons_norm)

        for r in self.rules:
            if normalize_entity_key(str(r["condition"])) == cond_key:
                old_cons = strip_accents(str(r["consequence"]).lower().strip())
                old_neg = old_cons.startswith("no ")
                old_core = normalize_entity_key(old_cons[3:] if old_neg else old_cons)
                if old_core == cons_core and old_neg != cons_neg:
                    return {
                        "conflict": True,
                        "existing_rule": r,
                        "message": (
                            f"⚠️ **Contradicción Lógica en Reglas Detectada:**\n"
                            f"• Regla previa: `Si {r['condition']} → {r['consequence']}`\n"
                            f"• Nueva afirmación opuesta: `Si {condition} → {consequence}`"
                        ),
                    }
        return None

    def add_rule(self, condition: str, consequence: str) -> Dict[str, object]:
        cond_key = normalize_entity_key(condition)
        for r in self.rules:
            if normalize_entity_key(str(r["condition"])) == cond_key:
                r["consequence"] = consequence.strip()
                return r

        rule = {
            "id": self.next_rule_id,
            "condition": condition.strip(),
            "consequence": consequence.strip(),
        }
        self.next_rule_id += 1
        self.rules.append(rule)
        return rule

    def check_fact_contradiction(
        self, candidate_fact: Dict[str, str], existing_facts: List[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        """
        Verifica si `candidate_fact` contradice hechos directos o propiedades heredadas
        en el grafo actual (ej. "A es B" + "B no puede C" vs "A puede C").
        """
        if candidate_fact.get("category") != "world":
            return None

        world_facts = [f for f in existing_facts if f.get("category") == "world"]
        if not world_facts:
            return None

        cand_subj = candidate_fact["subject"]
        cand_rel = candidate_fact["relation"]
        cand_obj = candidate_fact["object"]
        cand_subj_k = normalize_entity_key(cand_subj)
        cand_obj_k = normalize_entity_key(cand_obj)
        cand_pol, cand_fam = parse_relation_polarity(cand_rel)

        # 1. Contradicción directa sobre el mismo sujeto
        for f in world_facts:
            if (
                normalize_entity_key(str(f["subject"])) == cand_subj_k
                and normalize_entity_key(str(f["object"])) == cand_obj_k
            ):
                f_pol, f_fam = parse_relation_polarity(str(f["relation"]))
                if f_fam == cand_fam and f_pol != cand_pol:
                    return {
                        "conflict": True,
                        "steps": [f"{f['subject']} {f['relation']} {f['object']}"],
                        "message": (
                            f"⚠️ **Contradicción Lógica Directa Detectada:**\n"
                            f"• Mi grafo ya sabe que: **{f['subject']} {f['relation']} {f['object']}**.\n"
                            f"• Esto choca con tu afirmación: **{cand_subj} {cand_rel} {cand_obj}**."
                        ),
                    }

        # 2. Contradicción por herencia transitiva (simulando el grafo unido)
        simulated_facts = world_facts + [candidate_fact]
        subjects_to_check = []
        seen_s = set()
        for f in simulated_facts:
            sk = normalize_entity_key(str(f["subject"]))
            if sk and sk not in seen_s:
                seen_s.add(sk)
                subjects_to_check.append((str(f["subject"]), sk))

        for s_disp, s_key in subjects_to_check:
            direct_claims = [
                f for f in simulated_facts if normalize_entity_key(str(f["subject"])) == s_key
            ]
            deduced_claims = self.deduce_all_about_entity(s_disp, simulated_facts)

            # Mapa (familia_rel, obj_key) -> (polaridad, descripcion, pasos)
            claims_map: Dict[Tuple[str, str], Tuple[bool, str, List[str]]] = {}
            for df in direct_claims:
                pol, fam = parse_relation_polarity(str(df["relation"]))
                ok = normalize_entity_key(str(df["object"]))
                stmt = f"{str(df['subject']).capitalize()} {df['relation']} {df['object']}"
                claims_map[(fam, ok)] = (pol, stmt, [stmt])

            for ded in deduced_claims:
                pol, fam = parse_relation_polarity(str(ded["inferred_relation"]))
                ok = normalize_entity_key(str(ded["inferred_object"]))
                key = (fam, ok)
                if key in claims_map:
                    prev_pol, prev_stmt, prev_steps = claims_map[key]
                    if prev_pol != pol:
                        return {
                            "conflict": True,
                            "steps": ded["steps"],
                            "message": (
                                f"⚠️ **Conflicto Lógico por Inferencia Detectado sobre «{s_disp}»:**\n"
                                f"• Por cadena deductiva: {' ➔ '.join(ded['steps'])} (**{ded['conclusion']}**).\n"
                                f"• Pero se contradice con: **{prev_stmt}**."
                            ),
                        }
                else:
                    claims_map[key] = (pol, str(ded["conclusion"]), list(ded["steps"]))

        return None

    def deduce_all_about_entity(
        self, entity_text: str, facts: List[Dict[str, object]], max_hops: int = 4
    ) -> List[Dict[str, object]]:
        start_key = normalize_entity_key(entity_text)
        if not start_key:
            return []

        world_facts = [f for f in facts if f.get("category") == "world"]
        if len(world_facts) < 2:
            return []

        by_subj: Dict[str, List[Dict[str, object]]] = {}
        for f in world_facts:
            k = normalize_entity_key(str(f["subject"]))
            by_subj.setdefault(k, []).append(f)

        initial_facts = by_subj.get(start_key, [])
        if not initial_facts:
            return []

        root_display = str(initial_facts[0]["subject"]).strip()
        deductions: List[Dict[str, object]] = []
        visited = {start_key}

        queue: List[Tuple[str, List[Dict[str, object]]]] = []
        for f1 in initial_facts:
            pol1, fam1 = parse_relation_polarity(str(f1["relation"]))
            if pol1 and fam1 == "isa":
                next_key = normalize_entity_key(str(f1["object"]))
                if next_key and next_key not in visited:
                    queue.append((next_key, [f1]))

        while queue:
            curr_key, chain = queue.pop(0)
            if len(chain) >= max_hops:
                continue
            visited.add(curr_key)

            for nf in by_subj.get(curr_key, []):
                new_chain = chain + [nf]
                orig_rel = str(nf["relation"]).lower().strip()
                rel_nf = adapt_plural_to_singular_verb(orig_rel)
                obj_nf = singularize_word_if_needed(str(nf["object"]), orig_rel)

                steps_str = [
                    f"{str(step['subject']).capitalize()} {step['relation']} {step['object']}"
                    for step in new_chain
                ]
                conclusion = f"{root_display.capitalize()} {rel_nf} {obj_nf}"
                deductions.append(
                    {
                        "subject": root_display,
                        "inferred_relation": rel_nf,
                        "inferred_object": obj_nf,
                        "conclusion": conclusion,
                        "steps": steps_str,
                        "hops": len(new_chain),
                    }
                )

                pol_nf, fam_nf = parse_relation_polarity(orig_rel)
                if pol_nf and fam_nf == "isa":
                    nk = normalize_entity_key(str(nf["object"]))
                    if nk and nk not in visited:
                        queue.append((nk, new_chain))

        return deductions

    def backward_chain_prove(
        self,
        goal_subj: str,
        goal_rel: str,
        goal_obj: str,
        facts: List[Dict[str, object]],
        max_depth: int = 5,
    ) -> Optional[Dict[str, object]]:
        """
        Encadenamiento hacia atrás (Backward Chaining):
        Parte de la meta (X, rel, Y) y busca recursivamente las premisas y sub-objetivos
        que la demuestran, devolviendo el árbol completo de inferencia.
        """
        world_facts = [f for f in facts if f.get("category") == "world"]
        target_subj_k = normalize_entity_key(goal_subj)
        target_obj_k = normalize_entity_key(goal_obj)
        target_pol, target_fam = parse_relation_polarity(goal_rel)

        def _prove(s_key: str, fam: str, pol: bool, o_key: str, visited: Set[str], depth: int) -> Optional[Dict[str, object]]:
            if depth > max_depth:
                return None

            # 1. Caso Base: ¿Existe como axioma directo en el Grafo?
            for f in world_facts:
                fk_s = normalize_entity_key(str(f["subject"]))
                fk_o = normalize_entity_key(str(f["object"]))
                f_pol, f_fam = parse_relation_polarity(str(f["relation"]))
                if fk_s == s_key and fk_o == o_key and f_fam == fam and f_pol == pol:
                    stmt = f"{str(f['subject']).capitalize()} {f['relation']} {f['object']}"
                    return {"type": "axioma", "statement": stmt, "children": []}

            # 2. Reducción recursiva de meta: Buscar M tal que (s_key -> isa -> M) y probar (M, fam, pol, o_key)
            for f in world_facts:
                fk_s = normalize_entity_key(str(f["subject"]))
                f_pol, f_fam = parse_relation_polarity(str(f["relation"]))
                if fk_s == s_key and f_pol and f_fam == "isa":
                    mid_key = normalize_entity_key(str(f["object"]))
                    if not mid_key or mid_key in visited:
                        continue
                    sub_proof = _prove(mid_key, fam, pol, o_key, visited | {mid_key}, depth + 1)
                    if sub_proof is not None:
                        premise_stmt = f"{str(f['subject']).capitalize()} {f['relation']} {f['object']}"
                        goal_desc = f"Demostrado por herencia desde «{f['object']}»"
                        return {
                            "type": "inferencia",
                            "statement": goal_desc,
                            "children": [
                                {"type": "axioma", "statement": premise_stmt, "children": []},
                                sub_proof,
                            ],
                        }
            return None

        tree = _prove(target_subj_k, target_fam, target_pol, target_obj_k, {target_subj_k}, 0)
        return tree

    def _format_proof_tree(self, node: Dict[str, object], indent: int = 0) -> List[str]:
        prefix = "  " * indent + ("└─ " if indent > 0 else "🎯 ")
        tag = "[Axioma]" if node["type"] == "axioma" else "[Regla Silogística]"
        lines = [f"{prefix}{tag} {node['statement']}"]
        for child in node.get("children", []):
            lines.extend(self._format_proof_tree(child, indent + 1))
        return lines

    def answer_why_question(
        self, user_text: str, facts: List[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        """Responde preguntas tipo '¿Por qué X es/puede/no necesita Y?' usando Backward Chaining."""
        clean = re.sub(r"^[.!?¡¿\s]+|[.!?¡¿\s]+$", "", user_text.strip())
        m = re.match(
            r"(?i)^por\s+que\s+(?:el\s+|la\s+|los\s+|las\s+|un\s+|una\s+)?(.+?)\s+"
            r"(no\s+es\s+un|no\s+es\s+una|no\s+es|no\s+son|no\s+puede(?:n)?|no\s+necesita(?:n)?|no\s+tiene(?:n)?|"
            r"es\s+un|es\s+una|es|son|puede(?:n)?|necesita(?:n)?|tiene(?:n)?|vive(?:n)?\s+en|sirve(?:n)?\s+para|usa(?:n)?|come(?:n)?|hace(?:n)?)\s+(.+)$",
            strip_accents(clean),
        )
        if not m:
            return None

        subj_q = m.group(1).strip()
        rel_q = m.group(2).strip()
        obj_q = m.group(3).strip()

        proof_tree = self.backward_chain_prove(subj_q, rel_q, obj_q, facts)
        if proof_tree is not None:
            tree_lines = self._format_proof_tree(proof_tree)
            goal_title = f"{subj_q.capitalize()} {rel_q} {obj_q}"
            return {
                "conclusion": goal_title,
                "proof_tree": proof_tree,
                "reply": (
                    f"🧩 **Demostración por Backward Chaining para «{goal_title}»:**\n"
                    + "\n".join(tree_lines)
                ),
            }

        goal_cons_k = normalize_entity_key(f"{subj_q} {rel_q} {obj_q}")
        cond_chain = self._backward_chain_rules(goal_cons_k)
        if cond_chain:
            return {
                "conclusion": clean,
                "reply": (
                    f"🧩 **Árbol Backward Chaining (Modus Ponens Inverso):**\n"
                    + "\n".join(f"• {step}" for step in cond_chain)
                ),
            }
        return None

    def _backward_chain_rules(self, goal_key: str, max_depth: int = 4) -> List[str]:
        steps: List[str] = []
        curr_goal = goal_key
        visited = set()
        for _ in range(max_depth):
            matched = None
            for r in self.rules:
                rk_cons = normalize_entity_key(str(r["consequence"]))
                if rk_cons and (rk_cons == curr_goal or curr_goal in rk_cons) and r["id"] not in visited:
                    matched = r
                    break
            if not matched:
                break
            visited.add(matched["id"])
            steps.append(f"Porque se cumple la regla: Si {matched['condition']} ➔ entonces {matched['consequence']}")
            curr_goal = normalize_entity_key(str(matched["condition"]))
        return steps

    def evaluate_conditional_chain(self, user_text: str) -> Optional[Dict[str, object]]:
        if not self.rules:
            return None

        clean = re.sub(r"^[.!?¡¿\s]+|[.!?¡¿\s]+$", "", user_text.strip())
        m = re.match(r"(?i)^(?:que\s+(?:pasa|sucede|ocurre)\s+(?:si|cuando)|si)\s+(.+)$", strip_accents(clean))
        query_cond = m.group(1).strip() if m else clean
        curr_key = normalize_entity_key(query_cond)
        if not curr_key:
            return None

        chain_steps = []
        visited = set()
        final_consequence = None

        for _ in range(5):
            matched_rule = None
            padded_curr = f" {curr_key} "
            for r in self.rules:
                rk = normalize_entity_key(str(r["condition"]))
                if rk and (rk == curr_key or f" {rk} " in padded_curr) and r["id"] not in visited:
                    matched_rule = r
                    break
            if not matched_rule:
                break
            visited.add(matched_rule["id"])
            chain_steps.append(f"Si {matched_rule['condition']} → {matched_rule['consequence']}")
            final_consequence = str(matched_rule["consequence"])
            curr_key = normalize_entity_key(final_consequence)

        if chain_steps and final_consequence:
            return {
                "conclusion": final_consequence,
                "steps": chain_steps,
                "reply": (
                    f"🧩 **Deducción Lógica (Modus Ponens):** Por lo tanto, **{final_consequence}**.\n"
                    f"• Cadena razonada: {' ➔ '.join(chain_steps)}."
                ),
            }
        return None

    def answer_logical_question(
        self, user_text: str, facts: List[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        norm = re.sub(r"^[.!?¡¿\s]+|[.!?¡¿\s]+$", "", strip_accents(user_text.lower().strip()))

        if norm.startswith("por que "):
            why_res = self.answer_why_question(user_text, facts)
            if why_res is not None:
                return why_res
            after_why = re.sub(r"(?i)^por\s+que\s+", "", norm).strip()
            rule_back = self._backward_chain_rules(normalize_entity_key(after_why))
            if rule_back:
                return {
                    "conclusion": after_why,
                    "reply": "🧩 **Demostración Backward Chaining (Reglas):**\n" + "\n".join(f"• {s}" for s in rule_back),
                }

        cond_res = self.evaluate_conditional_chain(user_text)
        if cond_res is not None:
            return cond_res

        clean = re.sub(r"^[.!?¡¿\s]+|[.!?¡¿\s]+$", "", user_text.strip())
        q_stems = normalize_entity_key(clean)

        # 3. Inferencia multi-salto sobre entidades del grafo
        world_facts = [f for f in facts if f.get("category") == "world"]
        known_subjects = []
        seen_k = set()
        for f in world_facts:
            s = str(f["subject"])
            sk = normalize_entity_key(s)
            if sk and sk not in seen_k:
                seen_k.add(sk)
                known_subjects.append((s, sk))

        padded_q = f" {q_stems} "
        for subj_orig, subj_key in known_subjects:
            if subj_key and f" {subj_key} " in padded_q:
                deductions = self.deduce_all_about_entity(subj_orig, facts)
                if not deductions:
                    continue

                for ded in deductions:
                    obj_key = normalize_entity_key(str(ded["inferred_object"]))
                    rel_norm = strip_accents(str(ded["inferred_relation"]).lower())
                    if obj_key and f" {obj_key} " in padded_q:
                        is_negated = rel_norm.startswith("no ")
                        prefix = "No: " if (is_negated and "no " not in norm) else "Sí, por deducción lógica: "
                        return {
                            "conclusion": ded["conclusion"],
                            "steps": ded["steps"],
                            "reply": (
                                f"🧩 **{prefix}{ded['conclusion']}.**\n"
                                f"• Razonamiento paso a paso: {' ➔ '.join(ded['steps'])}."
                            ),
                        }

                if any(p in norm for p in ["deduc", "infer", "razon", "conclusion", "propiedad"]):
                    lines = [f"• **{d['conclusion']}** ({' ➔ '.join(d['steps'])})" for d in deductions[:4]]
                    return {
                        "conclusion": deductions[0]["conclusion"],
                        "steps": deductions[0]["steps"],
                        "reply": f"🧩 **Inferencias lógicas deducidas sobre «{subj_orig}»:**\n" + "\n".join(lines),
                    }

        return None

    def to_dict(self) -> dict:
        return {
            "next_rule_id": self.next_rule_id,
            "rules": self.rules,
        }

    def from_dict(self, data: dict):
        self.next_rule_id = int(data.get("next_rule_id", 1))
        self.rules = data.get("rules", [])
