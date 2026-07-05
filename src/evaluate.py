import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate trained machine learning models.

    Args:
        models (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        pd.DataFrame: Evaluation results.
    """

    results = []

    # ارزیابی هر مدل
    for name, model in models.items():

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
        )

        recall = recall_score(
            y_test,
            predictions,
        )

        f1 = f1_score(
            y_test,
            predictions,
        )

        roc_auc = roc_auc_score(
            y_test,
            predictions,
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)
    return results_df