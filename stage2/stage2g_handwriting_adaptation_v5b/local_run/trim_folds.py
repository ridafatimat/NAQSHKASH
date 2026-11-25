import json, pathlib

p = pathlib.Path(__file__).resolve().parent / "config.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
(p.parent / "config_before_trim.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

cfg["folds"]["configurations"].pop("cnn_bilstm")
cfg["folds"]["max_updates"] = 3000
p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

print("configurations:", list(cfg["folds"]["configurations"]))
print("max_updates:", cfg["folds"]["max_updates"])