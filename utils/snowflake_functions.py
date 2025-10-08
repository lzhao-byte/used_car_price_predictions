import snowflake.connector
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
import os
import polars as pl

def get_snow_session(database, schema, warehouse):
    user = os.environ.get("SNOWFLAKE_USER")
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    auth = os.environ.get("SNOWFLAKE_AUTHENTICATOR")
    try:
        current_session = get_active_session()
    except:
        conn_params = {
            'user': user,
            'account': account,
            'authenticator': auth,
            'database': database,
            'schema': schema,
            'warehouse': warehouse
        }
        current_session = Session.builder.configs(conn_params).create()
    return current_session


def fetch_data(snow_session=None, 
               table_name="used_cars",
               ref_name="make_model",
               words_name="stop_words",
               use_local=True):
    if use_local:
        try:
            df = pl.read_parquet("data/vehicles.parquet")
            df = df.with_columns(
                [pl.col(col).cast(pl.Float64) for col in df.columns if df[col].dtype in ([pl.Int64])]
            )
        except:
            df = pl.scan_parquet("data/**/*.parquet", hive_partitioning=True).collect()
            df = df.with_columns(
                [pl.col(col).cast(pl.Float64) for col in df.columns if df[col].dtype in ([pl.Int64])]
            )
        ref = pl.read_csv("data/make_model.csv")
        words = pl.read_csv("data/words.csv")
    else:
        df = snow_session.table(table_name).to_pandas()
        ref = snow_session.table(ref_name).to_pandas()
        words = snow_session.table(words_name).to_pandas()

        df = pl.from_pandas(df).rename(str.lower)
        ref = pl.from_pandas(df).rename(str.lower)
        words = pl.from_pandas(words).rename(str.lower)
    

    return df, ref, words







