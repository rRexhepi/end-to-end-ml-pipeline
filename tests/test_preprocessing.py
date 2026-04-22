import pandas as pd
import pytest

from preprocessing import FEATURES, Preprocessor


def test_fit_stores_training_stats(train_frame):
    pp = Preprocessor().fit(train_frame)
    assert pp.fitted
    assert pp.age_median == pytest.approx(train_frame["Age"].median())
    assert pp.embarked_mode == "S"


def test_transform_single_row_uses_training_median(train_frame):
    pp = Preprocessor().fit(train_frame)
    one_row = pd.DataFrame(
        [{
            "Pclass": 3, "Sex": "male", "Age": None,
            "SibSp": 0, "Parch": 0, "Fare": None, "Embarked": "S",
        }]
    )
    out = pp.transform(one_row)
    assert list(out.columns) == FEATURES
    assert len(out) == 1
    # Imputation happened — scaled Age is not NaN.
    assert out["Age"].notna().all()
    assert out["Fare"].notna().all()


def test_transform_rejects_missing_required_columns(train_frame):
    pp = Preprocessor().fit(train_frame)
    with pytest.raises(ValueError, match="Missing required columns"):
        pp.transform(pd.DataFrame([{"Pclass": 3, "Sex": "male"}]))


def test_transform_rejects_unknown_sex(train_frame):
    pp = Preprocessor().fit(train_frame)
    bad = pd.DataFrame(
        [{"Pclass": 1, "Sex": "unknown", "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 10.0, "Embarked": "S"}]
    )
    with pytest.raises(ValueError, match="Sex"):
        pp.transform(bad)


def test_transform_before_fit_raises(train_frame):
    pp = Preprocessor()
    with pytest.raises(RuntimeError):
        pp.transform(train_frame)


def test_unseen_embarked_falls_back_to_mode(train_frame):
    pp = Preprocessor().fit(train_frame)
    row = pd.DataFrame(
        [{"Pclass": 3, "Sex": "male", "Age": 20, "SibSp": 0, "Parch": 0, "Fare": 7.0, "Embarked": "X"}]
    )
    # Should not raise: falls back to the trained mode.
    out = pp.transform(row)
    assert len(out) == 1


def test_save_and_load_roundtrip(train_frame, tmp_path):
    pp = Preprocessor().fit(train_frame)
    path = tmp_path / "pp.pkl"
    pp.save(path)
    loaded = Preprocessor.load(path)
    pd.testing.assert_frame_equal(pp.transform(train_frame), loaded.transform(train_frame))
