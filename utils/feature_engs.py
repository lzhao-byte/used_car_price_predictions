from sklearn.cluster import KMeans
import plotly.express as px
import polars as pl
import numpy as np


class FeatureEng:
    df = None
    final = None

    def __init__(self, df):
        self.df = df
        self.final = df


    def show_samples(self, nsamples=10):
        return self.final.sample(nsamples)
    

    def get_columns(self):
        return self.final.columns


    def drop_columns(self, cols):
        self.final = self.final.drop(cols, strict=False)


    def select_columns(self, cols, target_col='price'):
        if target_col not in cols:
            cols += [target_col]
        self.final = self.final.select(cols)


    def _cluster_latlon(self, lat_col='lat', lon_col='long', ngroups=12, verbose=True):
        dfc = self.final
        coords = dfc.select(
            lat_col,
            lon_col,
        ).drop_nulls().unique().sort(
            lat_col,
            lon_col
        )
        km = KMeans(n_clusters=ngroups, init='k-means++', random_state=42).fit(coords)
        labels = km.labels_
        coords = coords.with_columns(group_region=labels)
        dfc = dfc.drop('group_region', strict=False).join(
            coords,
            on=[lat_col, lon_col]
        )
        return dfc
    


    def show_latlon_groups(self, lat_col='lat', lon_col='long', label_col='group_region'):
        fig = px.scatter(
                self.final,
                x=lon_col,
                y=lat_col,
                color=label_col,
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
        return fig
        
    
    def recode_condition(self):
        df = self.final
        conditions = {
            'other': 0,
            'salvage': 1,
            'fair': 2, 
            'good': 3, 
            'excellent': 4,
            'like new': 5, 
            'new': 6 
        }
        self.final = df.with_columns(
            pl.col("condition").replace(conditions, return_dtype=pl.Int64).alias('condition_num')
        )


    def add_features(self, add_age=False, add_annual_mileage=False, 
                     group_latlon=False, lat_col='lat', lon_col='long'):
        dfc = self.final
        if add_age:
            dfc = dfc.with_columns(
                pl.max_horizontal(pl.col('posting_date').str.to_datetime("%Y-%m-%dT%H:%M:%S%z").dt.year() - pl.col('year'), pl.lit(1)).alias('age'),
            )
        if add_annual_mileage:
            if 'age' not in dfc.columns:
                return "Age column is not present. Add age column first."
            dfc = dfc.with_columns(
                (pl.col('odometer')/pl.col('age')).alias('annual_mileage')
            )
        if group_latlon:
            assert lat_col and lon_col, "No valid lat/lon information for grouping."
            dfc = self._cluster_latlon(lat_col, lon_col)
        self.final = dfc
        

    def show_feature_dist(self, col):
        fig = px.histogram(
            self.clean,
            x = col,
            nbins=50
        )
        return fig


    def recat_target(self, target_col='price', target_cutoff=10000):
        dfc = self.clean
        dfc = dfc.with_columns(
            pl.when(
                pl.col(target_col) >= target_cutoff
            ).then(pl.lit("over"))
            .otherwise("below").alias("price_cutoff")
        )
