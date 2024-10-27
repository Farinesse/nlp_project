from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def make_model(model_type: str = "random_forest"):
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        ),
        "svc": SVC(
            kernel='linear',
            C=1.0,
            random_state=42
        ),
        "logistic": LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    if model_type not in models:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from: {list(models.keys())}")

    return models[model_type]
