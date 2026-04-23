from sklearn.metrics import accuracy_score, classification_report


def evaluate(model, X_test, y_test):
    """Print accuracy and full classification report."""
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
