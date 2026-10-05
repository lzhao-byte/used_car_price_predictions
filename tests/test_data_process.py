from datetime import datetime

import polars as pl
import pytest

from tests.conftest import DataPrep


def prep(**cols):
    return DataPrep(pl.DataFrame(cols))


class TestTrimFeature:
    def test_iqr_drops_extreme_outlier(self):
        p = prep(price=[float(x) for x in range(1000, 1010)] + [1_000_000.0])
        p.trim_feature("price", trim_method="IQR")
        assert 1_000_000.0 not in p.clean["price"].to_list()
        assert p.clean.height == 10

    def test_iqr_also_drops_non_positive_prices(self):
        p = prep(price=[0.0, -5.0, 1000.0, 1001.0, 1002.0, 1003.0])
        p.trim_feature("price", trim_method="IQR")
        assert p.clean["price"].min() > 0

    def test_percentile_trim_removes_both_tails(self):
        p = prep(price=[float(x) for x in range(1, 101)])
        p.trim_feature("price", trimprct=0.1)
        assert p.clean["price"].min() >= 10.9
        assert p.clean["price"].max() <= 90.1

    def test_fixed_bounds_are_inclusive(self):
        p = prep(price=[999.0, 1000.0, 50_000.0, 100_000.0, 100_001.0])
        p.trim_feature("price", left_end=1000, right_end=100_000)
        assert p.clean["price"].to_list() == [1000.0, 50_000.0, 100_000.0]


class TestTrimAge:
    def test_uses_posting_date_when_present(self):
        p = prep(
            year=[2022.0, 2021.0, 1990.0],
            posting_date=["2022-06-15T12:00:00+0000"] * 3,
        )
        p.trim_age(limits=30)
        # age 0 is excluded (needs >= 1); 1990 is 32 years old, over the limit
        assert p.clean["year"].to_list() == [2021.0]

    def test_accepts_datetime_typed_posting_date(self):
        p = prep(year=[2020.0, 1950.0], posting_date=[datetime(2022, 6, 15)] * 2)
        p.trim_age(limits=30)
        assert p.clean["year"].to_list() == [2020.0]

    def test_falls_back_to_current_year_without_posting_date(self):
        this_year = datetime.now().year
        p = prep(year=[float(this_year - 5), float(this_year - 50), float(this_year)])
        p.trim_age(limits=30)
        assert p.clean["year"].to_list() == [float(this_year - 5)]


class TestMissingValues:
    def test_remove_drops_rows_with_nulls_in_column(self):
        p = prep(a=[1, None, 3], b=[1, 2, 3])
        p.handle_nulls("a", method="Remove Null Values")
        assert p.clean["a"].to_list() == [1, 3]

    def test_drop_removes_the_column(self):
        p = prep(a=[1, None, 3], b=[1, 2, 3])
        p.handle_nulls("a", method="Drop Entire Column")
        assert p.clean.columns == ["b"]

    def test_fill_replaces_nulls_with_other(self):
        p = prep(fuel=["gas", None, "diesel"])
        p.handle_nulls("fuel", method='Fill with "Other"')
        assert p.clean["fuel"].to_list() == ["gas", "other", "diesel"]

    def test_get_nulls_reports_type_count_and_percent(self):
        p = prep(a=[1.0, None, None, 4.0])
        dtype, n_null, pct = p.get_nulls("a")
        assert dtype == pl.Float64
        assert n_null == 2
        assert pct == pytest.approx(50.0)

    def test_handle_nulls_all_applies_the_one_click_rules(self):
        p = prep(
            lat=[45.0, None, 46.0],
            long=[-122.0, -121.0, -120.0],
            year=[2015.0, 2016.0, 2017.0],
            manufacturer=["honda", "ford", "toyota"],
            model=["civic", "f150", "camry"],
            odometer=[10.0, 20.0, 30.0],
            VIN=["a", "b", "c"],
            fuel=[None, "gas", "gas"],
            cylinders=["4 cylinders", "6 cylinders", None],
            **{c: ["x", "y", "z"] for c in ("size", "transmission", "type", "drive", "title_status", "condition")},
        )
        p.handle_nulls_all()
        out = p.clean
        assert out.height == 2  # row with a null lat is dropped
        assert "VIN" not in out.columns
        assert out["fuel"].to_list() == ["other", "gas"]
        assert out["cylinders"].dtype == pl.Int64
        # the null is filled with the mode of the remaining rows
        assert out["cylinders"].to_list() == [4, 4]


class TestDuplicatesAndGeo:
    def test_remove_duplicates_keeps_one_row_per_key(self):
        p = prep(vin=["a", "a", "b"], price=[1, 2, 3])
        p.remove_duplicates(["vin"])
        assert sorted(p.clean["vin"].to_list()) == ["a", "b"]

    def test_identify_duplicates_returns_only_repeated_rows(self):
        p = prep(vin=["a", "a", "b"], price=[1, 1, 3])
        dupes = p.identify_duplicates(["vin", "price"])
        assert dupes.height == 2
        assert set(dupes["vin"]) == {"a"}

    def test_trim_latlon_keeps_continental_us(self):
        p = prep(
            lat=[45.5, 64.0, 0.0],  # Portland, Alaska, null island
            long=[-122.6, -150.0, 0.0],
        )
        p.trim_latlon()
        assert p.clean["lat"].to_list() == [45.5]


class TestSchemaHelpers:
    def test_get_cols_numeric_and_all(self):
        p = prep(n=[1.0], s=["x"], i=[1])
        assert sorted(p.get_cols("numeric")) == ["i", "n"]
        assert p.get_cols("all") == ["n", "s", "i"]

    def test_correct_types_parses_iso_timestamps(self):
        p = prep(posting_date=["2022-06-15T12:00:00+0000"])
        p.correct_types()
        assert p.clean["posting_date"].dtype == pl.Datetime

    def test_select_cols(self):
        p = prep(a=[1], b=[2], c=[3])
        p.select_cols(["a", "c"])
        assert p.clean.columns == ["a", "c"]

    def test_raw_is_preserved_after_cleaning(self):
        p = prep(a=[1, None, 3])
        p.handle_nulls("a", method="Remove Null Values")
        assert p.raw.height == 3
        assert p.clean.height == 2


class TestMakeModelStandardization:
    @pytest.fixture
    def reference(self):
        ref = pl.DataFrame(
            {
                "make": ["honda", "honda", "toyota", "ford", "land-rover"],
                "model": ["Civic (Sedan)", "Accord", "Camry LE", "F-150 XLT", "Range Rover"],
            }
        )
        words = pl.DataFrame({"words": ["xlt", "le", "lx"]})
        return ref, words

    def test_fuzzy_match_maps_messy_listings_to_reference(self, reference):
        ref, words = reference
        raw = pl.DataFrame(
            {
                "manufacturer": ["honda", "toyota", "ford", "rover"],
                "model": ["civic lx", "camry le", "f150 xlt crew cab", "range rover sport"],
            }
        )
        p = DataPrep(raw, ref=ref, words=words)
        p.clean_string_cols()
        # trailing whitespace in model_clean is covered separately in test_known_bugs.py
        got = {
            m: (make, clean.strip())
            for m, make, clean in zip(p.clean["model"], p.clean["make_clean"], p.clean["model_clean"])
        }
        assert got["civic lx"] == ("honda", "civic")
        assert got["camry le"] == ("toyota", "camry")
        assert got["f150 xlt crew cab"] == ("ford", "f-150")
        # "rover" is normalized to "land rover" before matching
        assert got["range rover sport"] == ("land rover", "range rover")

    def test_requires_manufacturer_and_model_columns(self, reference):
        ref, words = reference
        p = DataPrep(pl.DataFrame({"manufacturer": ["honda"]}), ref=ref, words=words)
        with pytest.raises(AssertionError, match="manufacturer and model"):
            p.clean_string_cols()
