import json
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.class_map import SONIC_CLASSES
from feature_extraction.features import FEATURE_VERSION

ROOT = Path(__file__).resolve().parents[1]


def scores(actual, predicted, classes):
    report = classification_report(
        actual, predicted, labels=classes, output_dict=True, zero_division=0
    )
    critical = [
        name
        for name in [
            "gunshot",
            "glass_breaking",
            "panic_scream",
            "aggression",
            "person_asking_for_help",
        ]
        if name in classes
    ]
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": report,
        "critical_class_recall": {name: report[name]["recall"] for name in critical},
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=classes
        ).tolist(),
        "classes": classes,
    }


def validate_dataset(source):
    if "feature_version" not in source or str(source["feature_version"]) != FEATURE_VERSION:
        raise ValueError("Outdated features. Rerun scripts/02_extract_features.py.")
    x, y, split = source["x"], source["y"], source["split"]
    if x.ndim != 2 or not (len(x) == len(y) == len(split)) or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix or mismatched labels/splits.")
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Training requires nonempty train, validation, and test splits.")
    classes = set(y)
    if len(classes) < 2 or not classes.issubset(SONIC_CLASSES):
        raise ValueError("Need at least two recognized classes.")
    for name in ("train", "validation", "test"):
        missing = classes - set(y[split == name])
        if missing:
            raise ValueError(f"{name} is missing classes after filtering: {sorted(missing)}")
    ids = source["audio_id"]
    if len(ids) != len(y):
        raise ValueError("Mismatched audio IDs.")
    groups = [set(ids[split == name]) for name in ("train", "validation", "test")]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Source recording leakage across dataset splits.")
    return x, y, split


def candidate_models():
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=7, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced", random_state=7, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=7),
    }
    for c in (1, 10, 100):
        for gamma in ("scale", 0.001):
            models[f"svm_c{c}_g{gamma}"] = make_pipeline(
                StandardScaler(),
                SVC(C=c, gamma=gamma, probability=False, class_weight="balanced", random_state=7),
            )
    return models


def main():
    with np.load(ROOT / "data" / "features.npz", allow_pickle=False) as source:
        x, y, split = validate_dataset(source)
    train = split == "train"
    valid = split == "validation"
    test = split == "test"
    classes = [name for name in SONIC_CLASSES if name in set(y)]
    models = candidate_models()
    results = {}
    for name, model in models.items():
        model.fit(x[train], y[train])
        validation = scores(y[valid], model.predict(x[valid]), classes)
        results[name] = validation
        (ROOT / "python_models" / f"{name}_metrics.json").write_text(
            json.dumps(
                {
                    "validation": validation,
                    "test": None,
                    "status": "validation complete; choose a model before final test",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(name, validation["accuracy"], validation["macro_f1"])
    selected = max(results, key=lambda name: results[name]["macro_f1"])
    model = models[selected]
    if selected.startswith("svm_"):
        model.set_params(svc__probability=True)
        model.fit(x[train], y[train])
    model.feature_version_ = FEATURE_VERSION
    joblib.dump(model, ROOT / "python_models" / f"{selected}.joblib")
    final = scores(y[test], model.predict(x[test]), classes)
    path = ROOT / "python_models" / f"{selected}_metrics.json"
    data = json.loads(path.read_text())
    data["test"] = final
    data["status"] = "final test evaluation complete"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (ROOT / "python_models" / "selection.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "selection_metric": "validation macro_f1",
                "feature_version": FEATURE_VERSION,
                "available_classes": classes,
                "missing_classes": [
                    name for name in SONIC_CLASSES if name not in classes
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Selected", selected, "test accuracy", final["accuracy"])


if __name__ == "__main__":
    main()
