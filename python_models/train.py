import json
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.class_map import SONIC_CLASSES

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
        "accuracy": report["accuracy"],
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


def main():
    source = np.load(ROOT / "data" / "features.npz", allow_pickle=True)
    x, y, split = source["x"], source["y"], source["split"]
    train = split == "train"
    valid = split == "validation"
    test = split == "test"
    classes = [name for name in SONIC_CLASSES if name in set(y)]
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=7, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=7),
        "svm": SVC(
            C=3, kernel="rbf", probability=True, class_weight="balanced", random_state=7
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(x[train], y[train])
        validation = scores(y[valid], model.predict(x[valid]), classes)
        results[name] = validation
        joblib.dump(model, ROOT / "python_models" / f"{name}.joblib")
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
    model = joblib.load(ROOT / "python_models" / f"{selected}.joblib")
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
