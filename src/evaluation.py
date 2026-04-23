from sklearn.metrics import accuracy_score, classification_report


def evaluate(model, X_test, y_test):
    """Print accuracy, AUC-ROC, and full classification report."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score
    print("Accuracy:", accuracy_score(y_test, preds))
    print("AUC-ROC: ", roc_auc_score(y_test, proba))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
