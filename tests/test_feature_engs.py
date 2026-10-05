import polars as pl

from tests.conftest import FeatureEng, make_listings


def test_recode_condition_orders_categories():
    fe = FeatureEng(pl.DataFrame({"condition": ["new", "salvage", "good", "other"]}))
    fe.recode_condition()
    assert fe.final["condition_num"].to_list() == [6.0, 1.0, 3.0, 0.0]


def test_select_columns_always_keeps_target():
    fe = FeatureEng(pl.DataFrame({"price": [1], "year": [2], "odometer": [3]}))
    fe.select_columns(["year"])
    assert set(fe.final.columns) == {"year", "price"}


def test_drop_columns_ignores_missing_names():
    fe = FeatureEng(pl.DataFrame({"a": [1], "b": [2]}))
    fe.drop_columns(["a", "does_not_exist"])
    assert fe.final.columns == ["b"]


class TestAddFeatures:
    def df(self):
        return pl.DataFrame(
            {
                "year": [2020.0, 2022.0],
                "odometer": [30_000.0, 5_000.0],
                "posting_date": ["2022-06-15T12:00:00+0000"] * 2,
            }
        )

    def test_age_is_at_least_one(self):
        fe = FeatureEng(self.df())
        fe.add_features(add_age=True)
        # 2022 - 2020 = 2; 2022 - 2022 = 0 is floored to 1
        assert fe.final["age"].to_list() == [2, 1]

    def test_annual_mileage_is_odometer_over_age(self):
        fe = FeatureEng(self.df())
        fe.add_features(add_age=True, add_annual_mileage=True)
        assert fe.final["annual_mileage"].to_list() == [15_000.0, 5_000.0]

    def test_annual_mileage_without_age_returns_hint_and_changes_nothing(self):
        fe = FeatureEng(self.df())
        msg = fe.add_features(add_annual_mileage=True)
        assert "Age column is not present" in msg
        assert "annual_mileage" not in fe.final.columns

    def test_group_latlon_assigns_one_region_per_location(self):
        fe = FeatureEng(make_listings(n=300))
        fe.add_features(group_latlon=True)
        out = fe.final
        assert "group_region" in out.columns
        assert out["group_region"].n_unique() == 12
        assert out["group_region"].null_count() == 0

    def test_group_latlon_is_deterministic(self):
        a = FeatureEng(make_listings(n=300))
        b = FeatureEng(make_listings(n=300))
        a.add_features(group_latlon=True)
        b.add_features(group_latlon=True)
        assert a.final.sort(["lat", "long"])["group_region"].to_list() == b.final.sort(["lat", "long"])["group_region"].to_list()


def test_show_samples_returns_requested_rows():
    fe = FeatureEng(pl.DataFrame({"a": list(range(20))}))
    assert fe.show_samples(5).height == 5


def test_latlon_groups_figure_plots_every_listing():
    fe = FeatureEng(make_listings(n=300))
    fe.add_features(group_latlon=True)
    fig = fe.show_latlon_groups()
    assert len(fig.data[0].x) == fe.final.height


class TestUsesWorkingFrame:
    """These methods used to raise AttributeError: they referenced self.clean, which FeatureEng does not have."""

    def test_show_feature_dist_plots_the_working_frame(self):
        fe = FeatureEng(pl.DataFrame({"price": [1.0, 2.0, 3.0]}))
        fig = fe.show_feature_dist("price")
        assert fig.data[0].type == "histogram"

    def test_recat_target_adds_the_cutoff_label(self):
        fe = FeatureEng(pl.DataFrame({"price": [5000.0, 20000.0]}))
        fe.recat_target(target_cutoff=10000)
        assert fe.final["price_cutoff"].to_list() == ["below", "over"]
