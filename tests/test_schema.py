import pandas as pd
import pytest
from housing_model.schema import validate_dataframe, SchemaError, REQUIRED_COLUMNS

def make_good_df():
    return pd.DataFrame([{
        "longitude": -122.0,
        "latitude": 37.0,
        "housing_median_age": 20,
        "total_rooms": 100,
        "total_bedrooms": 20,
        "population": 50,
        "households": 10,
        "median_income": 3.0,
        "ocean_proximity": "INLAND",
    }])

def test_validate_ok():
    df = make_good_df()
    out = validate_dataframe(df)
    assert list(out.columns) == list(df.columns)

def test_missing_column_fails():
    df = make_good_df().drop(columns=["median_income"])
    with pytest.raises(SchemaError):
        validate_dataframe(df)

def test_all_nan_numeric_fails():
    df = make_good_df()
    df["population"] = None
    with pytest.raises(SchemaError):
        validate_dataframe(df)
