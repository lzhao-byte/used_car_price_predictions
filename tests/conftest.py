"""Shared fixtures: small synthetic used-car listings (no network, no Snowflake)."""

import numpy as np
import polars as pl
import pytest

# The app classes are wrapped in st.cache_resource, which would hand every test
# the same cached instance. __wrapped__ is the plain class.
from utils.data_process import DataPrep as _DataPrep
from utils.feature_engs import FeatureEng as _FeatureEng
from utils.model_trains import ModelBuilder as _ModelBuilder

DataPrep = _DataPrep.__wrapped__
FeatureEng = _FeatureEng.__wrapped__
ModelBuilder = _ModelBuilder.__wrapped__


def make_listings(n=1300, seed=0):
    """Synthetic listings where price is (noisily) a function of age and mileage."""
    rng = np.random.default_rng(seed)
    year = rng.integers(2005, 2022, n)
    odometer = rng.integers(5_000, 200_000, n).astype(float)
    price = 30_000 - (2022 - year) * 900 - odometer * 0.06 + rng.normal(0, 500, n)
    return pl.DataFrame(
        {
            "price": price.clip(500).round(0),
            "year": year.astype(float),
            "odometer": odometer,
            "manufacturer": rng.choice(["toyota", "honda", "ford"], n),
            "condition": rng.choice(["good", "excellent", "fair", "like new"], n),
            "lat": rng.uniform(30, 47, n),
            "long": rng.uniform(-120, -75, n),
            "posting_date": ["2022-06-15T12:00:00+0000"] * n,
        }
    )


@pytest.fixture
def listings():
    return make_listings()


@pytest.fixture
def small_df():
    return pl.DataFrame(
        {
            "price": [1000.0, 5000.0, 9000.0, 12000.0, 15000.0],
            "year": [2010.0, 2012.0, 2015.0, 2018.0, 2020.0],
            "odometer": [150000.0, 120000.0, 80000.0, 40000.0, 10000.0],
        }
    )
