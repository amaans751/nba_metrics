import json, joblib, pathlib
from typing import Any, Dict

def save_artifacts(model, out_dir: str, feature_spec: Dict[str, Any] | None = None):
    p = pathlib.Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p / "model.pkl")
    if feature_spec is not None:
        (p / "feature_spec.json").write_text(json.dumps(feature_spec, indent=2))

def load_model(model_dir: str):
    return joblib.load(pathlib.Path(model_dir) / "model.pkl")
