import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def train_models(df: pd.DataFrame):

    # جدا کردن ویژگی‌ها از برچسب هدف
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    # تقسیم داده‌ها به Train و Temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # تقسیم Temporary به Validation و Test
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    # استانداردسازی داده‌ها
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    # مدل‌های مورد استفاده
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),

        "CatBoost": CatBoostClassifier(
            random_state=42,
            verbose=0,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
        ),
    }

    # Cross Validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = {}

    # آموزش مدل‌ها
    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
        )

        cv_scores[name] = scores.mean()

        model.fit(
            X_train,
            y_train,
        )

    return {
        "models": models,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
        "cv_scores": cv_scores,
    }