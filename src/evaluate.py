import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_models(
    models: dict,
    X_test,
    y_test,
):

    results = []
    confusion_matrices = {}

    for name, model in models.items():

        predictions = model.predict(X_test)

        # استخدام الاحتمالات لحساب ROC-AUC
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probabilities)
        else:
            roc_auc = float("nan")

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        confusion_matrices[name] = cm

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions),
                "Recall": recall_score(y_test, predictions),
                "F1 Score": f1_score(y_test, predictions),
                "ROC AUC": roc_auc,
            }
        )

    results_df = pd.DataFrame(results).round(4)

    return {
        "results": results_df,
        "confusion_matrices": confusion_matrices,
    }