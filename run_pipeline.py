from src.data_loader import load_data

from src.preprocessing import (
    preprocess_data,
    save_processed_data,
)

from src.features import (
    engineer_features,
    save_feature_dataset,
)

from src.mlflow_utils import (
    configure_mlflow,
    log_experiment,
)

from src.train import train_models
from src.evaluate import evaluate_models


def main():

    # تنظیم MLflow
    configure_mlflow()

    # بارگذاری نسخه اولیه دیتاست
    df = load_data("v1")

    # پیش‌پردازش داده‌ها
    processed_df = preprocess_data(df)

    # ذخیره نسخه دوم
    save_processed_data(processed_df)

    # استخراج ویژگی‌های جدید
    feature_df = engineer_features(processed_df)

    # ذخیره نسخه سوم
    save_feature_dataset(feature_df)

    # آموزش مدل‌ها
    training_output = train_models(feature_df)

    validation_output = evaluate_models(
        training_output["models"],
        training_output["X_validation"],
        training_output["y_validation"],
    )

    validation_results = validation_output["results"]

    best_model_name = validation_results.sort_values(
        by="Accuracy",
        ascending=False,
    ).iloc[0]["Model"]

    print("\nBest Model Selected:")
    print(best_model_name)

    evaluation_output = evaluate_models(
        {best_model_name: training_output["models"][best_model_name]},
        training_output["X_test"],
        training_output["y_test"],
    )

    results = evaluation_output["results"]
    confusion_matrices = evaluation_output["confusion_matrices"]

    print("\nFinal Test Results")
    print("-" * 60)
    print(results)

    # تسجيل أفضل نموذج فقط في MLflow
    row = results.iloc[0]

    metrics = {
        "accuracy": row["Accuracy"],
        "precision": row["Precision"],
        "recall": row["Recall"],
        "f1_score": row["F1 Score"],
        "roc_auc": row["ROC AUC"],
        "cv_accuracy": training_output["cv_scores"][best_model_name],
    }

    log_experiment(
        model_name=best_model_name,
        model=training_output["models"][best_model_name],
        metrics=metrics,
        dataset_version="v3",
        seed=42,
        confusion_matrix_data=confusion_matrices[best_model_name],
    )


if __name__ == "__main__":
    main()