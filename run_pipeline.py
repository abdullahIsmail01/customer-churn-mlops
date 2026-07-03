"""
Main pipeline entry point.
"""

from src.data_loader import load_data
from src.preprocessing import preprocess_data, save_processed_data


def main():
    # بارگذاری نسخه اولیه دیتاست
    df = load_data("v1")

    # پیش‌پردازش داده‌ها
    processed_df = preprocess_data(df)

    # ذخیره نسخه جدید دیتاست
    save_processed_data(processed_df)

    print("\nPreprocessing completed successfully.")
    print(f"Processed dataset shape: {processed_df.shape}")


if __name__ == "__main__":
    main()