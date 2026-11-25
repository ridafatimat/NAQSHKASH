import json, pathlib, hashlib

root = pathlib.Path(__file__).resolve().parent
gen = root.parent / "generated_only"

d = json.loads((gen / "decision.json").read_text(encoding="utf-8"))
assert d["selected_step"] == 750
ck = pathlib.Path(d["checkpoint"])
assert hashlib.sha256(ck.read_bytes()).hexdigest() == d["checkpoint_sha256"], "checkpoint hash mismatch"

d["passed"] = True
d["override"] = {
    "date": "2026-11-10",
    "decided_by": "user",
    "reason": "NAQSHKASH input is handwritten only. Start real-handwriting training from update 750 "
              "although clean synthetic gates failed (clean CER 2.84% vs 0.15% for frozen E1). "
              "Original gate result kept unchanged in decision.json.",
}
(gen / "decision_handwriting_override.json").write_text(json.dumps(d, indent=2), encoding="utf-8")

cfg_path = root / "config.json"
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
(root / "config_before_override.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
cfg["initialization"] = "step2"
cfg["generated_decision_path"] = "../generated_only/decision_handwriting_override.json"
cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

print("OK - folds will start from:", ck)