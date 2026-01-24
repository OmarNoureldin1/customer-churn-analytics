import pickle
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import preprocess_data


def evaluate():
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        "data/churn.csv"
    )

    model = pickle.load(open("models/churn_model.pkl", "rb"))

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report\n")
    print(classification_report(y_test, predictions))

    print("ROC-AUC Score:", roc_auc_score(y_test, probabilities))


if __name__ == "__main__":
    evaluate()
