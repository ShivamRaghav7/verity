"""Conservative structured workflow. This is NOT a trained language model."""

import re
from datetime import date
from decimal import Decimal
from time import perf_counter

RELATION_TERMS = {
    "device_limit": ["devices", "device", "connect", "connections"],
    "price_monthly": ["price", "cost", "monthly fee", "subscription fee"],
    "storage_gb": ["storage", "space", "gigabytes", "gb"],
    "support_hours": ["support hours", "support availability", "support available"],
}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12}


def mentions(text, term):
    return bool(re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", text))


def render_answer(fact, value, question):
    """Render only supported fact forms, explicitly naming subject and relation."""
    subject = f"{fact['entity_name']} {fact['plan']} {fact['version']}"
    if fact["relation"] == "device_limit":
        numeric = re.findall(r"\b(\d+)\s+devices?\b", question.lower())
        numeric += [str(v) for k, v in NUMBER_WORDS.items()
                    if re.search(r"\b" + k + r"\s+devices?\b", question.lower())]
        if re.search(r"\b(can|could|may)\b", question.lower()) and numeric:
            if len(set(numeric)) != 1:
                return None
            endorsement = "Yes" if Decimal(numeric[0]) <= Decimal(value) else "No"
            return f"{endorsement}. {subject} supports up to {value} devices."
        return f"{subject} supports up to {value} devices."
    if fact["relation"] == "price_monthly":
        return f"{subject} costs {value} {fact['unit']}."
    if fact["relation"] == "storage_gb":
        return f"{subject} includes {value} GB of storage."
    if fact["relation"] == "support_hours":
        return f"{subject} provides {value} {fact['unit']} of support."
    return None


class StructuredAssistant:
    """Immutable initial facts plus approved-memory resolution; no answer cache."""

    def __init__(self, store):
        self.store = store

    def ask(self, request):
        with self.store.snapshot():
            return self._ask(request)

    def _ask(self, request):
        started = perf_counter()
        result = {"question": request.question, "status": "clarification",
                  "answer": "Please specify one product, plan, version and property.",
                  "before_answer": None, "answer_kind": "deterministic_structured",
                  "route": "clarification", "fact": None,
                  "applicable_corrections": [], "revision": self.store.revision(),
                  "explanation": "Bounded structured parser; auxiliary models are not trained."}

        def finish(**updates):
            result.update(updates)
            result["latency_ms"] = round((perf_counter() - started) * 1000, 3)
            return result

        text = request.question.lower().strip()
        facts = self.store.facts()
        relations = [r for r, terms in RELATION_TERMS.items()
                     if any(mentions(text, term) for term in terms)]
        if len(relations) != 1:
            return finish(explanation="Ask about one supported property: devices, monthly price, storage, or support hours.")
        # This grammar is intentionally bounded. Unsupported wording must not
        # be silently collapsed into a confident template answer.
        allowed = set("how many much what is are does do can could may i my we our it the a an s on with under for of per up to allowed allowance limit limits support supports supported connect use link device devices connections price cost costs monthly fee subscription storage space included includes gigabytes gb hours day availability available plan version".split())
        for fact in facts:
            for phrase in [fact["entity_id"], fact["entity_name"], fact["plan"], fact["version"], *fact["aliases"]]:
                allowed.update(re.findall(r"[a-z0-9]+", phrase.lower()))
        allowed.update(NUMBER_WORDS)
        tokens = re.findall(r"[a-z0-9]+", text)
        if any(token not in allowed and not token.isdigit() and not re.fullmatch(r"v\d+", token) for token in tokens):
            return finish(answer="Please rephrase as one direct factual question, or select an explicit scope.",
                          explanation="Unsupported wording in the bounded structured demo; no learned model has interpreted it.")
        if re.search(r"\b(can|could|may)\b", text):
            number = r"(?:\d+|" + "|".join(NUMBER_WORDS) + r")"
            if not re.match(r"^(?:can|could|may) i (?:connect|use|link) " + number + r" devices? (?:on|with|under) ", text):
                return finish(explanation="Supported capacity form: Can I connect four devices on Basic?")
        if any(mentions(text, w) for w in ["used to", "previously", "older", "before", "originally"]) and not request.as_of and not re.search(r"\bv(?:ersion\s*)?\d+\b", text):
            return finish(answer="Which historical version or effective date do you mean?",
                          explanation="A historical request needs an explicit version or date.")

        # Prefer the longest product alias to avoid Nimbus matching Nimbus Mini.
        alias_hits = {(match.start(), match.end(), f["entity_id"])
                      for f in facts
                      for a in [f["entity_name"], f["entity_id"], *f["aliases"]]
                      for match in re.finditer(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", text)}
        products = {entity for start, end, entity in alias_hits
                    if not any(s <= start and end <= e and (s, e) != (start, end)
                               for s, e, _ in alias_hits)}
        plans = {f["plan"] for f in facts if mentions(text, f["plan"])}
        versions = {"v" + m for m in re.findall(r"\bv(?:ersion\s*)?(\d+)\b", text)}
        dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
        if len(products) > 1 or len(plans) > 1 or len(versions) > 1 or len(dates) > 1:
            return finish(explanation="Multiple scopes were mentioned; ask about one scope at a time.")
        for field, found in [("entity_id", products), ("plan", plans), ("version", versions), ("as_of", dates)]:
            chosen = getattr(request, field, None)
            if chosen and found and str(chosen) not in found:
                return finish(answer=f"The selected {field.replace('_', ' ')} conflicts with your question.",
                              explanation="Selectors cannot silently override explicit question scope.")
        product = request.entity_id or next(iter(products), None)
        plan = request.plan or next(iter(plans), None)
        version = request.version or next(iter(versions), None)
        as_of = str(request.as_of or next(iter(dates), date.today().isoformat()))
        try:
            date.fromisoformat(as_of)
        except ValueError:
            return finish(answer="Please provide a valid ISO date.")
        candidates = [f for f in facts if f["relation"] == relations[0]
                      and (not product or f["entity_id"] == product)
                      and (not plan or f["plan"] == plan)
                      and (not version or f["version"] == version)
                      and f["effective_from"] <= as_of
                      and (not f.get("effective_to") or as_of < f["effective_to"])]
        if not version and candidates:
            scopes = {(f["entity_id"], f["plan"]) for f in candidates}
            if len(scopes) == 1:
                # Current version follows the catalogue, never a guessed scope.
                latest = max(candidates, key=lambda f: (f["effective_from"], int(re.sub(r"\D", "", f["version"]) or "0")))
                candidates = [f for f in candidates if f["version"] == latest["version"]]
        if len(candidates) != 1:
            return finish(explanation="No unique catalogue scope matches. Select the product, plan and version.")
        fact = candidates[0]
        state = self.store.resolve_fact(fact["fact_id"], as_of)
        before = render_answer(fact, fact["value"], text)
        if state["conflict"]:
            return finish(status="conflict", answer="An overlapping correction is unresolved. An administrator must resolve it.",
                          before_answer=before, fact=fact, route="conflict",
                          applicable_corrections=state["edits"], revision=state["revision"],
                          explanation="Conflicting values are withheld; no latest-write-wins rule is used.")
        answer = render_answer(fact, state["value"], text)
        if answer is None:
            return finish(explanation="This question form needs clarification.")
        return finish(status="answered", answer=answer, before_answer=before,
                      fact=fact, route="approved_correction" if state["edits"] else "initial_catalogue",
                      applicable_corrections=state["edits"], revision=state["revision"],
                      explanation="Structured answer from the scoped fact store. No neural answer was generated.")
