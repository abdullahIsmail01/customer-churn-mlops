from src.data_loader import load_data
from src.preprocessing import (
    preprocess_data,
    save_processed_data,
)
from src.features import (
    engineer_features,
    save_feature_dataset,
)


def main():
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

    print("\nPipeline completed successfully.")
    print(f"Final dataset shape: {feature_df.shape}")


if __name__ == "__main__":
    main()