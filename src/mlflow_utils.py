import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.catboost
from catboost import CatBoostClassifier

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
    dataset_version: str,
    seed: int,
    confusion_matrix_data,
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

        mlflow.log_param(
            "dataset_version",
            dataset_version,
        )

        mlflow.log_param(
            "seed",
            seed,
        )

        # ذخیره مدل
        try:

            if isinstance(model, XGBClassifier):

                mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="model",
                )

            elif isinstance(model, CatBoostClassifier):

                mlflow.catboost.log_model(
                    cb_model=model,
                    name="model",
                )

            else:

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                )

            # حفظ Confusion Matrix كصورة
            plt.figure(figsize=(5, 4))

            sns.heatmap(
                confusion_matrix_data,
                annot=True,
                fmt="d",
                cmap="Blues",
            )

            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(model_name)

            figure_name = f"{model_name}_cm.png"

            plt.savefig(figure_name)
            plt.close()

            mlflow.log_artifact(figure_name)

        except Exception as error:

            print(f"\nWarning: Model '{model_name}' was not saved.")
            print(error)