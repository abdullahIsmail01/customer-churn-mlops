"""
Main pipeline entry point.
"""

from src.data_loader import load_data, show_dataset_info


def main():
    # بارگذاری نسخه اولیه داده
    df = load_data("v1")

    # نمایش اطلاعات اولیه دیتاست
    show_dataset_info(df)


if __name__ == "__main__":
    main()