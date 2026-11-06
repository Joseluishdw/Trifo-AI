"""
Motor de Razonamiento e Inferencia Lógica desde Cero (Symbolic Reasoning Engine).
Capacidades:
1. Silogismos e Inferencia Transitiva Multi-Salto (Encadenamiento hacia adelante y atrás):
   Ej: "Python es un lenguaje interpretado" + "Los lenguajes interpretados no necesitan compilarse"
   => Deduce: "Python no necesita compilarse" mostrando la cadena lógica paso a paso.
2. Modus Ponens sobre Reglas Condicionales ("Si A entonces B" + "Si B entonces C").
3. Unificación Morfológica (conecta singular/plural como "humano" <-> "humanos").
"""

import re
from typing import List, Dict, Optional, Tuple
from .tokenizer import strip_accents, spanish_stem


def normalize_entity_key(text: str) -> str:
    """
    Normaliza una entidad eliminando artículos y reduciendo cada palabra a su raíz
    morfológica para unificar singular y plural (ej: 'un humano' == 'los humanos').
    """
    t = strip_accents(text.lower().strip())
    t = re.sub(r"[^\w\sñ]", " ", t)
    words = [w for w in t.split() if w not in {"el", "la", "los", "las", "un", "una", "unos", "unas", "todo", "todos", "toda", "todas"}]
    if not words:
        return t.strip()
    stems = [spanish_stem(w) for w in words]
    return " ".join(stems)


def adapt_plural_to_singular_verb(rel: str) -> str:
    mapping = {
        "son": "es",
        "tienen": "tiene",
        "pueden": "puede",
        "necesitan": "necesita",
        "no necesitan": "no necesita",
        "no pueden": "no puede",
        "viven en": "vive en",
        "comen": "come",
        "usan": "usa",
        "hacen": "hace",
        "sirven para": "sirve para",
    }
    return mapping.get(rel.lower().strip(), rel)


class LogicalReasoningEngine:
    """
    Motor de inferencia simbólica que opera sobre los hechos del Grafo de Conocimiento
    y sobre reglas condicionales aprendidas en el chat.
    """

    def __init__(self):
        # Lista de reglas condicionales: [{"id", "condition", "consequence"}]
        self.rules: List[Dict[str, object]] = []
        self.next_rule_id: int = 1

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

    def deduce_all_about_entity(
        self, entity_text: str, facts: List[Dict[str, object]], max_hops: int = 4
    ) -> List[Dict[str, object]]:
        """
        Dado un sujeto (ej. 'Sócrates' o 'Python'), recorre el grafo de hechos buscando
        cadenas deductivas de herencia ('es' / 'es un' / 'pertenece a') de hasta max_hops saltos.
        Retorna deducciones de >= 2 saltos (conocimiento inferido no dicho directamente).
        """
        start_key = normalize_entity_key(entity_text)
        if not start_key:
            return []

        world_facts = [f for f in facts if f.get("category") == "world"]
        if len(world_facts) < 2:
            return []

        # Construir índice por clave normalizada del sujeto
        by_subj: Dict[str, List[Dict[str, object]]] = {}
        for f in world_facts:
            k = normalize_entity_key(str(f["subject"]))
            by_subj.setdefault(k, []).append(f)

        # Cola BFS: (current_key, display_subject_inicial, chain_steps)
        isa_relations = {"es", "es un", "es una", "es el", "es la", "son", "pertenece a"}
        deductions: List[Dict[str, object]] = []
        visited = {start_key}

        # Encontrar el nombre original del sujeto inicial
        initial_facts = by_subj.get(start_key, [])
        if not initial_facts:
            return []

        root_display = str(initial_facts[0]["subject"]).strip()

        queue: List[Tuple[str, List[Dict[str, object]]]] = []
        for f1 in initial_facts:
            rel1 = str(f1["relation"]).lower().strip()
            if rel1 in isa_relations:
                next_key = normalize_entity_key(str(f1["object"]))
                if next_key and next_key not in visited:
                    queue.append((next_key, [f1]))

        while queue:
            curr_key, chain = queue.pop(0)
            if len(chain) >= max_hops:
                continue
            visited.add(curr_key)

            next_facts = by_subj.get(curr_key, [])
            for nf in next_facts:
                new_chain = chain + [nf]
                orig_rel = str(nf["relation"]).lower().strip()
                rel_nf = adapt_plural_to_singular_verb(orig_rel)
                obj_nf = str(nf["object"]).strip()
                if orig_rel == "son" and " " not in obj_nf:
                    if obj_nf.endswith("es") and len(obj_nf) > 4 and obj_nf[-3] not in ("a", "e", "i", "o", "u"):
                        obj_nf = obj_nf[:-2]
                    elif obj_nf.endswith("s") and len(obj_nf) > 3:
                        obj_nf = obj_nf[:-1]

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

                if str(nf["relation"]).lower().strip() in isa_relations:
                    nk = normalize_entity_key(obj_nf)
                    if nk and nk not in visited:
                        queue.append((nk, new_chain))

        return deductions

    def evaluate_conditional_chain(self, user_text: str) -> Optional[Dict[str, object]]:
        """
        Evalúa Modus Ponens: si el usuario pregunta "¿Qué pasa si X?" o afirma "X",
        encadena las reglas condicionales Si X -> Y, Si Y -> Z.
        """
        if not self.rules:
            return None

        clean = re.sub(r"[.!?¡¿]+$", "", user_text.strip())
        # Extraer núcleo si pregunta "que pasa si..." / "entonces que pasa cuando..."
        m = re.match(r"(?i)^(?:que\s+(?:pasa|sucede|ocurre)\s+(?:si|cuando)|si)\s+(.+)$", clean)
        query_cond = m.group(1).strip() if m else clean
        curr_key = normalize_entity_key(query_cond)
        if not curr_key:
            return None

        chain_steps = []
        visited = set()
        final_consequence = None

        for _ in range(5):
            matched_rule = None
            for r in self.rules:
                rk = normalize_entity_key(str(r["condition"]))
                if rk and (rk == curr_key or (len(rk) >= 4 and rk in curr_key)) and r["id"] not in visited:
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
        """
        Intenta responder una pregunta usando inferencia lógica multi-salto sobre el grafo
        o encadenamiento de reglas condicionales.
        """
        # 1. Revisar encadenamiento de reglas condicionales ("¿Qué pasa si llueve?")
        cond_res = self.evaluate_conditional_chain(user_text)
        if cond_res is not None:
            return cond_res

        clean = re.sub(r"[.!?¡¿]+$", "", user_text.strip())
        norm = strip_accents(clean.lower())

        # 2. Buscar entidades conocidas mencionadas en la pregunta y calcular sus deducciones
        world_facts = [f for f in facts if f.get("category") == "world"]
        known_subjects = []
        seen_k = set()
        for f in world_facts:
            s = str(f["subject"])
            sk = normalize_entity_key(s)
            if sk and sk not in seen_k:
                seen_k.add(sk)
                known_subjects.append((s, sk))

        q_stems = normalize_entity_key(clean)

        for subj_orig, subj_key in known_subjects:
            if subj_key and subj_key in q_stems:
                deductions = self.deduce_all_about_entity(subj_orig, facts)
                if not deductions:
                    continue

                # Verificar si la pregunta consulta específicamente por una propiedad deducida
                # Ej: "¿Sócrates es mortal?" o "¿Python necesita compilarse?"
                for ded in deductions:
                    obj_key = normalize_entity_key(str(ded["inferred_object"]))
                    rel_norm = strip_accents(str(ded["inferred_relation"]).lower())
                    if obj_key and obj_key in q_stems:
                        is_negated = "no " in rel_norm
                        prefix = "No: " if (is_negated and "no " not in norm) else "Sí, por deducción lógica: "
                        return {
                            "conclusion": ded["conclusion"],
                            "steps": ded["steps"],
                            "reply": (
                                f"🧩 **{prefix}{ded['conclusion']}.**\n"
                                f"• Razonamiento paso a paso: {' ➔ '.join(ded['steps'])}."
                            ),
                        }

                # Si pregunta "¿qué puedes deducir de X?" o "¿por qué...?" o pregunta general sobre X
                if any(p in norm for p in ["deduc", "infer", "razon", "conclusion", "por que", "propiedad"]):
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
