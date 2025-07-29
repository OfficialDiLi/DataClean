"""Unit tests for DataService."""

import pandas as pd
from src.services.data_service import DataService


def test_load_file_csv(tmp_path):
    # Create a sample CSV file
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("a,b\n1,2\n3,4")
    with open(csv_file, "rb") as f:
        df = DataService.load_file(f)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_clean_data():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, "x", "y"]})
    cleaned = DataService.clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
