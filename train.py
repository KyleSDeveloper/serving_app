import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

# Optional MLflow logging (will no-op if mlflow not installed)
def try_mlflow_log(params: dict, metrics: dict, artifact_path: str, model_path: str):
    try:
        import mlflow
        mlflow.set_experiment("ml_serving_pipeline")
        with mlflow.start_run():
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path=artifact_path)
    except Exception as e:
        print(f"[train] MLflow logging skipped: {e}")

def main():
    X, y = load_iris(return_X_y=True, as_frame=False)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params = {"n_estimators": 150, "max_depth": 5, "random_state": 42}
    clf = RandomForestClassifier(**params)
    clf.fit(Xtr, ytr)

    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"[train] test_accuracy={acc:.3f}")

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.pkl"
    joblib.dump(clf, out_path)
    print(f"[train] saved model to {out_path.resolve()}")

    try_mlflow_log(params=params, metrics={"test_accuracy": acc}, artifact_path="artifacts", model_path=str(out_path))

if __name__ == "__main__":
    main()
