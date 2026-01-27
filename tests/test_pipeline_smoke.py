import pandas as pd
from housing_model.pipeline import make_pipeline

def test_pipeline_fit_predict_smoke():
    df = pd.DataFrame({
        "longitude": [-122.0, -121.0],
        "latitude": [37.0, 36.0],
        "housing_median_age": [20, 30],
        "total_rooms": [100, 200],
        "total_bedrooms": [20, 40],
        "population": [50, 80],
        "households": [10, 20],
        "median_income": [3.0, 4.0],
        "ocean_proximity": ["NEAR BAY", "INLAND"],
    })
    y = pd.Series([100000, 150000])

    pipe = make_pipeline(random_state=42, n_jobs=1)
    pipe.fit(df, y)
    pred = pipe.predict(df)
    assert len(pred) == 2