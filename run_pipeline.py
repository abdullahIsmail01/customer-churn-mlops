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

    # ارزیابی مدل‌ها
    results = evaluate_models(
        training_output["models"],
        training_output["X_test"],
        training_output["y_test"],
    )

    print("\nModel Evaluation Results")
    print("-" * 60)
    print(results)

    # ثبت نتایج در MLflow
    for _, row in results.iterrows():

        model_name = row["Model"]

        metrics = {
            "accuracy": row["Accuracy"],
            "precision": row["Precision"],
            "recall": row["Recall"],
            "f1_score": row["F1 Score"],
            "roc_auc": row["ROC AUC"],
        }

        log_experiment(
            model_name=model_name,
            model=training_output["models"][model_name],
            metrics=metrics,
        )


if __name__ == "__main__":
    main()