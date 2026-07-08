from pathlib import Path

import pandas as pd


def load_data(version: str = "v1") -> pd.DataFrame:

    # مسیر فایل داده بر اساس نسخه انتخاب شده
    data_path = Path("data") / version / "Telco_customer_churn.xlsx"

    # بررسی وجود فایل
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # خواندن فایل اکسل
    df = pd.read_excel(data_path)

    return df


def show_dataset_info(df: pd.DataFrame) -> None:

    print("\n========== Dataset Information ==========\n")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes.to_string())

    print("\nMissing Values:")
    print(df.isnull().sum().to_string())

    print("\nFirst Five Rows:")
    print(df.head().to_string())