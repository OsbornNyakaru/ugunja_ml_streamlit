import pandas as pd

from ugunja_app.schema import validate_dataframe


def test_missing_columns_errors():
    df = pd.DataFrame({"a": [1]})
    result = validate_dataframe(df, "Test", required_columns={"a", "b"})
    assert result.errors
    assert any("missing columns" in error for error in result.errors)


def test_empty_dataframe_errors():
    df = pd.DataFrame(columns=["a"])
    result = validate_dataframe(df, "Test", required_columns={"a"})
    assert any("dataset is empty" in error for error in result.errors)


def test_invalid_dates_warning():
    df = pd.DataFrame({"date": [pd.NaT], "value": [1]})
    result = validate_dataframe(
        df,
        "Test",
        required_columns={"date", "value"},
        date_columns=["date"],
    )
    assert any("invalid or missing dates" in warning for warning in result.warnings)


def test_non_numeric_warning():
    df = pd.DataFrame({"value": ["x", "2"]})
    result = validate_dataframe(
        df,
        "Test",
        required_columns={"value"},
        numeric_columns=["value"],
    )
    assert any("non-numeric" in warning for warning in result.warnings)


def test_valid_dataframe():
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "value": [1]})
    result = validate_dataframe(
        df,
        "Test",
        required_columns={"date", "value"},
        date_columns=["date"],
        numeric_columns=["value"],
    )
    assert not result.errors
    assert not result.warnings
