import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

def train_models(df: pd.DataFrame):

    # جدا کردن ویژگی‌ها از برچسب هدف
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    # تقسیم داده‌ها به آموزش و آزمون
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

# استانداردسازی داده‌ها
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
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
        "X_test": X_test,
        "y_test": y_test,
        "cv_scores": cv_scores,
    }