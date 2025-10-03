import polars as pl
import plotly.express as px
from rapidfuzz import process, fuzz
import re
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
from polars import selectors as ps
import numpy as np


class DataPrep:
    raw = None
    clean = None
    ref = None
    words = None

    def __init__(self, df, ref=None, words=None):
        self.raw = df
        self.clean = df
        self.ref = ref
        self.words = words
    
    def show_schema(self):
        return pl.DataFrame(self.clean.schema)
    

    def show_sample_values(self):
        cols = [col for col in self.clean.columns 
                if self.clean.select(pl.col(col)).count().item() != 0]
        return self.clean.drop_nulls(subset=cols).head(5)
    

    def show_sample_col_values(self, col=None):
        if col is None:
            cols = [col for col in self.clean.columns if self.clean[col].dtype not in (pl.Int64, pl.Float64, pl.Float32, pl.Datetime)]
            col = random.choice(cols)
        try:
            samples = self.clean[col].value_counts().sort(by='count', descending=True).sample(25).to_series()
        except:
            samples = self.clean[col].value_counts().sort(by='count', descending=True).to_series()
        return "Column **" + col.title() + "**\n\n" + " || ".join(samples)


    def select_cols(self, cols):
        self.clean = self.clean.select(cols)


    def get_cols(self, dtype='numeric'):
        if dtype == 'numeric':
            cols = [col for col in self.clean.columns if self.clean[col].dtype in (pl.Int64, pl.Float64, pl.Float32)]
        elif dtype == 'date':
            cols = [col for col in self.clean.columns if self.clean[col].dtype in (pl.Datetime)]
        elif dtype == 'string':
            cols = [col for col in self.clean.columns if self.clean[col].dtype in (pl.String)]
        else:
            cols = self.clean.columns
        return cols


    def show_feature_dist(self, col):
        fig = px.histogram(
            self.clean,
            x = col,
            nbins=50
        )
        return fig
    

    def get_nulls(self, col):
        datatype = self.clean[col].dtype
        null_counts = self.clean[col].null_count()
        counts = self.clean[col].count()
        null_perc = null_counts / (null_counts + counts) * 100
        return datatype, null_counts, null_perc


    def trim_feature(self, col='price', 
                    trimprct=None, 
                    left_end=1000,
                    right_end=100000,
                    trim_method=None):
        df = self.clean
        if trim_method == 'IQR':
            dfc = df.filter(
                pl.col(col).is_between(
                    pl.col(col).quantile(0.25) - 1.5 * (pl.col(col).quantile(0.75)-pl.col(col).quantile(0.25)), 
                    pl.col(col).quantile(0.75) + 1.5 * (pl.col(col).quantile(0.75)-pl.col(col).quantile(0.25)), 
                ),
                pl.col(col) > 0
            )

        elif trimprct is not None:
            dfc = df.filter(
                pl.col(col).is_between(
                    pl.col(col).quantile(trimprct), 
                            pl.col(col).quantile(1-trimprct)
                ),
                pl.col(col) > 0
            )
        else:
            dfc = df.filter(
                    pl.col(col).is_between(left_end, right_end)
                )
        
        self.clean = dfc

    def trim_age(self, limits=30):
        df = self.clean
        if 'posting_date' in df.columns:
            try:
                dfc = df.filter(
                    (pl.col('posting_date').dt.year() - pl.col('year')).is_between(1, limits)
                )
            except:
                dfc = df.filter(
                    (pl.col('posting_date').str.to_datetime("%Y-%m-%dT%H:%M:%S%z").dt.year() - pl.col('year')).is_between(1, limits)
                )
        else:
            dfc = df.filter(
                    (datetime.now().year - pl.col('year')).is_between(1, limits)
                )
        self.clean = dfc

    def correct_types(self, dt_col='posting_date', from_data_type="String", to_data_type="Date"):
        df = self.clean

        if from_data_type=='String' and to_data_type=='Date':
            try:
                dfc = df.with_columns(
                    pl.col(dt_col).str.to_datetime("%Y-%m-%dT%H:%M:%S%z")
                )
                self.clean = dfc
            except:
                pass

        if from_data_type=='Number' and to_data_type=='String':
            try:
                dfc = df.with_columns(
                    pl.col(dt_col).cast(pl.String)
                )
                self.clean = dfc
            except:
                pass




    def handle_nulls(self, col, target_col='price', method='remove_null'):
        df = self.clean
        if 'remove' in method.lower():
            dfc = df.drop_nulls(subset=col)
        elif 'drop' in method.lower():
            dfc = df.drop(col)
        elif 'fill' in method.lower():
            if col == 'type':
                dfc = df.with_columns(
                            pl.col('model').map_elements(
                                lambda x: re.findall(r'\b(?:awd|4wd|fwd|rwd)\b', x, re.IGNORECASE)[0].replace("awd", "4wd"), return_dtype=pl.String
                            ).alias("drive_from_model")
                    ).with_columns(
                        pl.when(pl.col(col).is_null())
                        .then(pl.col('drive_from_model'))
                        .otherwise(pl.col(col)).alias(pl.col(col))
                    ).with_columns(pl.col(col).fill_null('other'))
            else:
                dfc = df.with_columns(pl.col(col).fill_null('other'))
        elif 'impute' in method.lower():
            if 'regression' in method.lower():
                subset = [ocol for ocol in df.columns if ocol not in (col, target_col)]
                df_impute = df.filter(
                    pl.col(col).is_not_null()
                ).with_columns(
                    [pl.col(scol).fill_null(strategy=pl.col(scol).median()) 
                     if df[scol].dtype in (pl.Int64, pl.Float64, pl.Int32)
                     else pl.col(scol).fill_null(pl.col(scol).drop_nulls().mode())
                     for scol in subset]
                )
                rf = RandomForestRegressor() if df[col].dtype in (pl.Int64, pl.Float64, pl.Int32) else RandomForestClassifier()
                rf.fit(df_impute.select(subset), df_impute.select(col))
                rf.predict()
                dfc = df.with_columns(
                    pl.col(col).fill_null(rf.predict())
                )
            else:
                strategy = pl.col(col).fill_null(strategy=pl.col(col).median()) if df[col].dtype in (pl.Int64, pl.Float64, pl.Float32) else pl.col(col).fill_null(pl.col(col).drop_nulls().mode())
                dfc = df.with_columns(strategy)
        else:
            dfc = df
        self.clean = dfc



    def handle_nulls_all(self):
        df = self.clean
        dropna_cols = ['lat', 'long', 'year', 'manufacturer', 'model', 'odometer']
        drop_cols = ['VIN', 'url', 'image_url', 'region_url', 'paint_color', 'description', 'county']
        fill_other_cols = ['fuel', 'size', 'transmission', 'type', 'drive', 'title_status', 'condition']

        dfc = df.drop_nulls(
            subset=dropna_cols
            ).drop(
                drop_cols, strict=False
            ).with_columns(
                [pl.col(col).fill_null("other") for col in fill_other_cols]
            ).with_columns(
                pl.col('cylinders').str.replace(" cylinders", "").alias('cylinders')
            ).with_columns(
                pl.when(pl.col('cylinders')!='other').then(pl.col('cylinders'))
                .fill_null(pl.col('cylinders').drop_nulls().mode())
                .cast(pl.Int64).alias('cylinders')
            )
       
        self.clean = dfc



    def identify_duplicates(self, cols):
        return self.clean.filter(self.clean.select(cols).is_duplicated()).sort(cols)



    def remove_duplicates(self, cols, verbose=True):
        df = self.clean
        dfc = df.unique(subset=cols)
        self.clean = dfc

        if verbose:
            return  f"Number of records before duplicates: {df.select(pl.len())}.\nNumber of records after duplicates: {dfc.select(pl.len())}."


    def draw_latlon(self, lat_col='lat', lon_col='long'):
        fig = px.scatter(
            self.clean,
            x = lon_col,
            y = lat_col
        )
        return fig


    def trim_latlon(self, lat_col='lat', lon_col='long'):
        df = self.clean
        dfc = df.filter(
            pl.col(lat_col).is_between(24, 49),
            pl.col(lon_col).is_between(-125, -66)
        )
        self.clean = dfc


    def clean_string_cols(self):
        dfc = self._clean_model_columns()
        self.clean = dfc


    def _clean_region_columns(self):
        ## normalize state name/abb
        df = self.clean
        dfc = df.with_columns(
            pl.col('region')
            .str.to_lowercase()
            .str.replace(" / ", "-")
        )
        return dfc

    def _clean_model_columns(self):
        df = self.clean

        assert 'manufacturer' in df.columns and 'model' in df.columns, "Make sure you have manufacturer and model columns in the raw data."
        if self.ref is not None:
            assert 'model' in self.ref.columns and 'make' in self.ref.columns, "Make sure you have make and model in ref data."

        dfc = df.with_columns(
            pl.col("manufacturer").replace("rover", "land rover").replace("datsun", 'nissan').str.replace("-", " "),
        )

        if self.ref is not None:
            ### reference make model
            non_essentials = "|".join(sorted(self.words['words'].unique()))
            refx = self.ref.with_columns(
                    pl.col('model')
                    .str.replace(r"\s*\([^)]*\)", "")
                    .str.to_lowercase()
                    .str.replace(f'({non_essentials})', '')
                    .alias("model_cl"),
                    pl.col("make").str.replace("-", " ").alias("make"),
                ).with_columns(
                    (pl.col("make") + " " + pl.col("model_cl").str.split(" ").list.head(2).list.join(" ")).str.to_lowercase().alias("make_model")
                ).select('make_model').unique()
            ref_models = refx['make_model'].to_list()

            ### create a make - model column for better matching
            dfc = dfc.with_columns(
                    pl.col("model").str.replace_all(r"[^a-zA-Z0-9 ]", "")
                    .str.replace(r'\s*(crew|regular|extended|double|super|quad|mega|club|xtra|access)\s*cab\b', '')
                    .str.replace(f'({non_essentials})', '')
                    .alias("model_cl")
            ).with_columns(
                (pl.col("manufacturer") + " " + pl.col("model_cl").str.split(" ").list.head(3).list.join(" ")).str.to_lowercase().alias("make_model")
            )
            raw_models = dfc['make_model'].unique().to_list()

            ### create a lookup table for raw <-> matched model
            model_lookup = {model: process.extractOne(model, ref_models, scorer=fuzz.QRatio)[0] if process.extractOne(model, ref_models, scorer=fuzz.QRatio) is not None else "" for model in raw_models}

            ### fuzzy match
            dfc = dfc.with_columns(
                pl.col("make_model").replace_strict(model_lookup).alias("matched_model")
            )

            ## attach confidence score for checking
            dfc = dfc.with_columns(
                (pl.struct("make_model", 'matched_model').map_elements(lambda x: fuzz.QRatio(x['make_model'], x['matched_model']), return_dtype=pl.Float32)).alias("confidence")
            )

            ## for those unmatched manufacturers
            dfc = dfc.with_columns(
                pl.when(
                        pl.col("matched_model").str.split(" ").list.slice(0,1).list.join("") != pl.col('manufacturer'),
                        pl.col('manufacturer').str.split(" ").list.len() != 2
                    ).then(pl.col("model").str.split(" ").list.slice(0, 2).list.join(" "))
                    .when(
                        pl.col('manufacturer').str.split(" ").list.len() == 2
                    ).then(pl.col("matched_model").str.split(" ").list.slice(2,2).list.join(" "))
                    .otherwise(pl.col("matched_model").str.split(" ").list.slice(1,2).list.join(" ")).alias("model_clean"),
                    pl.col("manufacturer").alias("make_clean")
            ).drop(
                'model_cl', 'matched_model', 'make_model', 'confidence'
            )

        return dfc


        
