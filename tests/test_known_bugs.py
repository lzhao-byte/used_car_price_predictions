"""Regression tests for bugs found while writing the suite (see git log)."""

import polars as pl
import pytest

from tests.conftest import DataPrep


class TestImputeMissingValues:
    """The 'Impute with Mode/Median' button on the Data Prep page used to raise TypeError."""

    def test_numeric_column_gets_the_median(self):
        p = DataPrep(pl.DataFrame({"a": [1.0, None, 3.0, 100.0]}))
        p.handle_nulls("a", method="Impute with Mode/Median")
        assert p.clean["a"].to_list() == [1.0, 3.0, 3.0, 100.0]

    def test_string_column_gets_the_mode(self):
        p = DataPrep(pl.DataFrame({"fuel": ["gas", "gas", None, "diesel"]}))
        p.handle_nulls("fuel", method="Impute with Mode/Median")
        assert p.clean["fuel"].to_list() == ["gas", "gas", "gas", "diesel"]

    def test_regression_imputation_is_not_implemented(self):
        p = DataPrep(pl.DataFrame({"a": [1.0, None, 3.0], "price": [1, 2, 3]}))
        with pytest.raises(NotImplementedError):
            p.handle_nulls("a", method="impute regression")


class TestFillDriveType:
    """Filling nulls in 'type' used to crash (invalid alias; IndexError when no drive token)."""

    def test_uses_drive_token_from_model_when_present(self):
        p = DataPrep(pl.DataFrame({"model": ["rav4 awd", "camry"], "type": [None, None]}))
        p.handle_nulls("type", method='Fill with "Other"')
        assert p.clean["type"].to_list() == ["4wd", "other"]  # awd is normalized to 4wd

    def test_existing_values_are_kept(self):
        p = DataPrep(pl.DataFrame({"model": ["f150 4wd"], "type": ["truck"]}))
        p.handle_nulls("type", method='Fill with "Other"')
        assert p.clean["type"].to_list() == ["truck"]

    def test_helper_column_is_not_left_behind(self):
        p = DataPrep(pl.DataFrame({"model": ["x"], "type": [None]}))
        p.handle_nulls("type", method='Fill with "Other"')
        assert "drive_from_model" not in p.clean.columns


class TestGetCols:
    def test_string_columns(self):
        p = DataPrep(pl.DataFrame({"n": [1.0], "s": ["x"]}))
        assert p.get_cols("string") == ["s"]

    def test_date_columns(self):
        from datetime import datetime

        p = DataPrep(pl.DataFrame({"n": [1.0], "d": [datetime(2022, 1, 1)]}))
        assert p.get_cols("date") == ["d"]


class TestModelCleanWhitespace:
    """Stop-word removal left trailing spaces, so 'camry' and 'camry ' counted as different models."""

    def test_model_clean_has_no_surrounding_whitespace(self):
        ref = pl.DataFrame({"make": ["toyota", "ford"], "model": ["Camry LE", "F-150 XLT"]})
        words = pl.DataFrame({"words": ["xlt", "le"]})
        raw = pl.DataFrame(
            {"manufacturer": ["toyota", "ford"], "model": ["camry le", "f150 xlt"]}
        )
        p = DataPrep(raw, ref=ref, words=words)
        p.clean_string_cols()
        assert p.clean["model_clean"].to_list() == ["camry", "f-150"]
