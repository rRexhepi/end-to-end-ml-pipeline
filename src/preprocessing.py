"""Preprocessing for the Titanic dataset.

`Preprocessor` is a small stateful transformer in the sklearn style:
`fit` learns imputation values, encoders, and scalers from the TRAINING
set; `transform` applies them to any frame (train, val, test, or a single
inference row).

Why this matters: the previous version used `df['Age'].fillna(df['Age'].median())`
at inference time. On a single-row request that's the row's own age — or NaN
if the value is missing. We now save the training median and reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES: list[str] = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "SibSp",
    "Parch",
]

REQUIRED_INPUT_COLUMNS: list[str] = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

NUMERICAL_FEATURES: list[str] = ["Age", "Fare", "FamilySize"]


@dataclass
class Preprocessor:
    age_median: float = 0.0
    fare_median: float = 0.0
    embarked_mode: str = "S"
    embarked_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self._require_columns(df)
        self.age_median = float(df["Age"].median())
        self.fare_median = float(df["Fare"].median())
        self.embarked_mode = str(df["Embarked"].mode().iloc[0])

        filled = df.copy()
        filled["Embarked"] = filled["Embarked"].fillna(self.embarked_mode)
        self.embarked_encoder.fit(filled["Embarked"])

        engineered = self._engineer(filled)
        self.scaler.fit(engineered[NUMERICAL_FEATURES])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor.transform called before fit().")
        self._require_columns(df)
        out = df.copy()

        out["Age"] = out["Age"].fillna(self.age_median)
        out["Fare"] = out["Fare"].fillna(self.fare_median)
        out["Embarked"] = out["Embarked"].fillna(self.embarked_mode)

        # Unseen Embarked categories would crash LabelEncoder; fall back to the mode.
        known = set(self.embarked_encoder.classes_)
        out["Embarked"] = out["Embarked"].where(out["Embarked"].isin(known), self.embarked_mode)
        out["Embarked"] = self.embarked_encoder.transform(out["Embarked"])

        out["Sex"] = out["Sex"].map({"male": 0, "female": 1})
        # Map returns NaN for unknown values; surface that clearly.
        if out["Sex"].isna().any():
            raise ValueError("Sex must be 'male' or 'female'.")

        out = self._engineer(out)
        out[NUMERICAL_FEATURES] = self.scaler.transform(out[NUMERICAL_FEATURES])
        return out[FEATURES].copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)

    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["FamilySize"] = df["SibSp"].fillna(0).astype(int) + df["Parch"].fillna(0).astype(int) + 1
        df["IsAlone"] = np.where(df["FamilySize"] > 1, 0, 1)
        return df

    @staticmethod
    def _require_columns(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
