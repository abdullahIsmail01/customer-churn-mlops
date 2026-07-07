import mlflow
import mlflow.sklearn


def configure_mlflow():


    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        experiment_name="Customer Churn Prediction"
    )


def log_experiment(
    model_name: str,
    model,
    metrics: dict,
):


    with mlflow.start_run(
        run_name=model_name,
    ):

        # ثبت پارامترهای مدل
        mlflow.log_params(
            model.get_params()
        )

        # ثبت معیارهای ارزیابی
        mlflow.log_metrics(
            metrics
        )

        # ذخیره مدل
        try:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
            )

        except Exception as error:

            print(f"\nWarning: Model '{model_name}' was not saved.")
            print(error)