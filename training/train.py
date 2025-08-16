from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import json, pathlib

def main():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)

    models = pathlib.Path("models")
    models.mkdir(parents=True, exist_ok=True)
    dump(clf, models / "model.pkl")
    (models / "meta.json").write_text(json.dumps({"n_features": X.shape[1]}), encoding="utf-8")
    print("✅ Saved model to models/model.pkl and meta.json")

if __name__ == "__main__":
    main()
