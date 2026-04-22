import pandas as pd
import pytest


@pytest.fixture
def train_frame() -> pd.DataFrame:
    """A minimal, stand-in Titanic-shaped frame that's enough to fit a Preprocessor."""
    return pd.DataFrame(
        {
            "PassengerId": range(1, 11),
            "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
            "Name": ["A"] * 10,
            "Sex": ["male", "female", "female", "female", "male",
                    "male", "male", "female", "female", "male"],
            "Age": [22, 38, 26, 35, 35, None, 54, 2, 27, 14],
            "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
            "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
            "Ticket": ["T"] * 10,
            "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07],
            "Cabin": [None] * 10,
            "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C"],
        }
    )
