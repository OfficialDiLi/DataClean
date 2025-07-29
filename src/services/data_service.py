"""Data service for handling data operations."""

from typing import Optional
import pandas as pd
from pathlib import Path
from src.utils.logging_utils import logger
from src.config.config import ALLOWED_EXTENSIONS


class DataService:
    @staticmethod
    def load_file(file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file."""
        try:
            file_extension = Path(file.name).suffix.lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                logger.error(f"Unsupported file type: {file_extension}")
                return None

            if file_extension == ".csv":
                return pd.read_csv(file)
            elif file_extension in [".xlsx", ".xls"]:
                return pd.read_excel(file)
            elif file_extension == ".ods":
                return pd.read_excel(file, engine="odf")
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return None

        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return None

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning operations."""
        # Remove duplicate rows
        df = df.drop_duplicates()

        # Handle missing values
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Fill numeric columns with median
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Fill categorical columns with mode
        df[categorical_cols] = df[categorical_cols].fillna(
            df[categorical_cols].mode().iloc[0]
        )

        return df

    @staticmethod
    def get_data_profile(df: pd.DataFrame) -> dict:
        """Generate a data profile report."""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }
