import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocess import preprocess_data


def train_all_models():
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        "data/churn.csv"
    )

    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        pickle.dump(
            model,
            open(f"models/{name}_model.pkl", "wb")
        )

    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    print("✅ All models trained and saved")


if __name__ == "__main__":
    train_all_models()
