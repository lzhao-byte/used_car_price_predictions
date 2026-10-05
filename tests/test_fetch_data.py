"""fetch_data's local path: partitioned Parquet plus the two reference CSVs."""

import polars as pl

from utils.snowflake_functions import fetch_data

load = fetch_data.__wrapped__  # skip st.cache_resource


def write_data_dir(root):
    data = root / "data"
    (data / "price_quantile_bin=(0, 2500]").mkdir(parents=True)
    (data / "price_quantile_bin=(2500, inf]").mkdir(parents=True)
    pl.DataFrame({"price": [1000.0, 2000.0], "year": [2005.0, 2008.0]}).write_parquet(
        data / "price_quantile_bin=(0, 2500]" / "0.parquet"
    )
    pl.DataFrame({"price": [9000.0], "year": [2015.0]}).write_parquet(
        data / "price_quantile_bin=(2500, inf]" / "0.parquet"
    )
    pl.DataFrame({"make": ["honda"], "model": ["Civic"]}).write_csv(data / "make_model.csv")
    pl.DataFrame({"words": ["xlt"]}).write_csv(data / "words.csv")


def test_reads_all_partitions_and_drops_the_partition_column(tmp_path, monkeypatch):
    write_data_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    df, ref, words = load(use_local=True)
    assert df.height == 3
    assert "price_quantile_bin" not in df.columns
    assert ref.columns == ["make", "model"]
    assert words["words"].to_list() == ["xlt"]


def test_prefers_single_file_when_present(tmp_path, monkeypatch):
    write_data_dir(tmp_path)
    pl.DataFrame({"price": [1.0], "year": [2000.0], "price_quantile_bin": ["x"]}).write_parquet(
        tmp_path / "data" / "vehicles.parquet"
    )
    monkeypatch.chdir(tmp_path)
    df, _, _ = load(use_local=True)
    assert df.height == 1
    assert "price_quantile_bin" not in df.columns
