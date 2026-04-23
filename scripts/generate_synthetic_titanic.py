"""Generate a synthetic Titanic-shaped dataset when Kaggle data isn't available.

Writes ``data/train.csv`` and ``data/test.csv`` with the same columns and
rough distributions as Kaggle's classic Titanic competition, so the
training + serving pipeline can be exercised end-to-end without Kaggle
credentials. This is **not** a replacement for the real dataset — the
model trained on it is a toy — but it's enough to exercise the
Preprocessor, MLflow registry, drift monitor, and FastAPI/metrics path.

Why the sampling logic: we encode the same correlations the real dataset
has (women + first class survive at higher rates) so the trained model
makes qualitatively sensible predictions. Reviewers who dig in will see
the correlations are hand-wired; that's fine — the dataset is explicitly
synthetic.

Usage:
    python scripts/generate_synthetic_titanic.py --rows 891 --test-rows 418
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_DEFAULT = 891
TEST_DEFAULT = 418


def _build_frame(n: int, *, with_target: bool, start_id: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = rng.normal(loc=29, scale=14, size=n).clip(0, 80)
    # Kaggle has ~20% missing Age.
    age = np.where(rng.random(n) < 0.2, np.nan, age)
    sibsp = rng.integers(0, 5, size=n)
    parch = rng.integers(0, 3, size=n)
    fare = np.where(
        pclass == 1,
        rng.gamma(3.0, 25.0, size=n),
        np.where(pclass == 2, rng.gamma(2.0, 10.0, size=n), rng.gamma(2.0, 5.0, size=n)),
    )
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

    frame = pd.DataFrame(
        {
            "PassengerId": np.arange(start_id, start_id + n),
            "Pclass": pclass,
            "Name": [f"Synthetic, {'Mrs.' if s == 'female' else 'Mr.'} {i}" for i, s in enumerate(sex)],
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Ticket": ["SYN" + str(i) for i in range(n)],
            "Fare": fare,
            "Cabin": [None] * n,
            "Embarked": embarked,
        }
    )

    if with_target:
        # Survival probability: women and 1st-class passengers survive more,
        # matching the real-world correlations so the model learns something
        # besides noise.
        base = 0.15
        p = base
        p = p + np.where(sex == "female", 0.55, 0.0)
        p = p + np.where(pclass == 1, 0.15, 0.0)
        p = p + np.where(pclass == 3, -0.08, 0.0)
        p = p + np.where(np.nan_to_num(age, nan=30) < 10, 0.15, 0.0)
        p = np.clip(p, 0.01, 0.99)
        frame["Survived"] = (rng.random(n) < p).astype(int)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=TRAIN_DEFAULT, help="Training rows")
    parser.add_argument("--test-rows", type=int, default=TEST_DEFAULT, help="Test rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = _build_frame(args.rows, with_target=True, start_id=1, seed=args.seed)
    test = _build_frame(args.test_rows, with_target=False, start_id=args.rows + 1, seed=args.seed + 1)

    train_path = args.out_dir / "train.csv"
    test_path = args.out_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Wrote synthetic training data: {train_path} ({len(train)} rows)")
    print(f"Wrote synthetic test data:     {test_path} ({len(test)} rows)")
    print(
        "\nThese are synthetic samples — the model trained on them is a toy. "
        "Replace with Kaggle Titanic CSVs for a real experiment."
    )


if __name__ == "__main__":
    main()
