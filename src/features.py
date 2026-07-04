from pathlib import Path

import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    # ایجاد ویژگی داشتن خانواده
    df["Has Family"] = (
        (df["Partner_Yes"] == 1) |
        (df["Dependents_Yes"] == 1)
    ).astype(int)

    # تعداد سرویس‌های حفاظتی
    protection_columns = [
        "Online Security_Yes",
        "Online Backup_Yes",
        "Device Protection_Yes",
        "Tech Support_Yes",
    ]

    df["Protection Services Count"] = df[
        protection_columns
    ].sum(axis=1)

    # تعداد سرویس‌های سرگرمی
    streaming_columns = [
        "Streaming TV_Yes",
        "Streaming Movies_Yes",
    ]

    df["Streaming Services Count"] = df[
        streaming_columns
    ].sum(axis=1)

    # میانگین هزینه ماهانه
    df["Average Monthly Spending"] = (
        df["Total Charges"] /
        df["Tenure Months"].replace(0, np.nan)
    )

    df["Average Monthly Spending"] = (
        df["Average Monthly Spending"]
        .fillna(df["Monthly Charges"])
    )

    # مشتری قدیمی یا جدید
    df["Is Long-Term Customer"] = (
        df["Tenure Months"] >= 24
    ).astype(int)

    return df


def save_feature_dataset(
    df: pd.DataFrame,
    version: str = "v3",
) -> None:
    """
    Save dataset after feature engineering.
    """

    output_dir = Path("data") / version

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        "Telco_customer_churn_features.xlsx"
    )

    # ذخیره نسخه جدید دیتاست
    df.to_excel(
        output_path,
        index=False,
    )