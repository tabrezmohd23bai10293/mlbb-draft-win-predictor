import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import load_and_prepare_data


def train_model(
    csv_path: str = "data/drafts.csv",
    model_out: str = "models/mlbb_win_model.pkl"
):
    X, y, vectorizer, df = load_and_prepare_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("MLBB Draft Win Predictor - Training Complete")
    print("=" * 60)
    print(f"Total samples      : {len(df)}")
    print(f"Train samples      : {len(X_train)}")
    print(f"Test samples       : {len(X_test)}")
    print(f"Accuracy           : {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["red", "blue"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        model_out
    )

    print(f"\nSaved model to: {model_out}")


if __name__ == "__main__":
    train_model()
