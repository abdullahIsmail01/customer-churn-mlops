from pathlib import Path

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """

    # ستون‌هایی که برای آموزش مدل مناسب نیستند
    columns_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
    ]

    # حذف ستون‌های غیرضروری
    df = df.drop(
    columns=columns_to_drop,
    errors="ignore",
    )
    # تبدیل مقادیر نامعتبر به مقدار خالی
    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce",
    )

    # حذف رکوردهایی که مقدار Total Charges ندارند
    df = df.dropna(subset=["Total Charges"])

    # استخراج ستون‌های متنی
    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    # حذف ستون هدف از لیست
    if "Churn Value" in categorical_columns:
        categorical_columns.remove("Churn Value")

    # تبدیل ویژگی‌های متنی به ویژگی‌های عددی
    df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int,
    ) 

    return df


def save_processed_data(
    df: pd.DataFrame,
    version: str = "v2",
) -> None:
    """
    Save the processed dataset.
    """

    output_dir = Path("data") / version

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "Telco_customer_churn_preprocessed.xlsx"

    # ذخیره نسخه جدید دیتاست
    df.to_excel(
        output_path,
        index=False,
    )