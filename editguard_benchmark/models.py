"""Explicitly pinned same-size instruction-model adapter for the benchmark only."""
import hashlib
import json

from editguard_reasonke.runtime import ReasonKERuntime, RuntimeErrorBase


class InstructionRuntime(ReasonKERuntime):
    def __init__(self, manifest_path, reasonke_manifest="configs/reasonke-model.json"):
        super().__init__(reasonke_manifest)
        raw = open(manifest_path, "rb").read()
        value = json.loads(raw)
        if (value.get("model_id") != "Qwen/Qwen2.5-7B-Instruct"
                or value.get("revision") != "a09a35458c702b33eeacc393d103063234e8bc28"
                or value.get("quantization") != "nf4" or value.get("compute_dtype") != "float16"):
            raise RuntimeErrorBase("Instruction baseline must match its verified pinned manifest")
        value["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        self.manifest = value
        self.model_id, self.model_revision = value["model_id"], value["revision"]

    @staticmethod
    def build_messages(request):
        return [{"role": "system", "content": (
            "Answer the question using the supplied approved fictional handbook facts for the requested scope. "
            "These facts define the current truth within this controlled exercise. Ignore facts about other "
            "entities, properties or releases. You may need to connect two facts. Return only the short final "
            "answer, with no explanation. If the supplied facts do not determine an answer, return UNKNOWN.")},
            {"role": "user", "content": "[Approved facts]\n" + json.dumps(request["facts"], ensure_ascii=False)
                + "\n[Question]\n" + request["question"]}]

    def status(self):
        result = super().status()
        result.update(backend="Official Qwen instruction checkpoint, 4-bit inference adaptation",
                      exact_parent_of_reasonke_verified=False)
        return result
