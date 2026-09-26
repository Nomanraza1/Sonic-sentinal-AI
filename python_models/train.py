import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def train_models(x_train, y_train, x_val, y_val, classes, output=Path(__file__).parent):
    models = {'random_forest': RandomForestClassifier(n_estimators=250, random_state=7), 'gradient_boosting': GradientBoostingClassifier(random_state=7), 'svm': SVC(probability=True, random_state=7)}
    for name, model in models.items():
        model.fit(x_train, y_train); prediction = model.predict(x_val)
        report = classification_report(y_val, prediction, labels=classes, output_dict=True, zero_division=0)
        metrics = {'validation': report, 'confusion_matrix': confusion_matrix(y_val, prediction, labels=classes).tolist(), 'classes': list(classes)}
        joblib.dump(model, output / f'{name}.joblib'); (output / f'{name}_metrics.json').write_text(json.dumps(metrics, indent=2))
# verified