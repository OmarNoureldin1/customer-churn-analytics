import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data


def train():
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        "data/churn.csv"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model and scaler
    pickle.dump(model, open("models/churn_model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    print("✅ Model training completed and saved.")


if __name__ == "__main__":
    train()
