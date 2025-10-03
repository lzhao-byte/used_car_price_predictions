import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from itertools import combinations

class DataExp:
    df = None

    def __init__(self, df):
        self.df = df
    

    def describe(self):
        return self.df.describe()


    def show_nulls(self):
        df_nulls = self.df.describe().transpose(
            column_names='statistic', 
            include_header=True
        ).with_columns(
            pl.col('null_count').cast(pl.Float32),
            pl.col('count').cast(pl.Float32),
        ).with_columns(
            (pl.col('null_count') / (pl.col('count')+pl.col('null_count')) * 100).alias('Null_%')
        ).sort('Null_%')

        fig = px.bar(
            df_nulls,
            x = 'column',
            y = 'Null_%',
            labels={"Null_%": '%Nulls', 'column': 'Feature'}
        )
        # fig.update_layout(height=500)
        return fig
    

    
    def check_target_dist(self, target_col='price', boxplot=True, trimplot=True, trimprct=None, 
                    left_end=1000,
                    right_end=100000
                    ):
        fig1, fig2 = None, None
        if boxplot:
            fig1 = px.box(
                self.df,
                y = target_col
            )
        if trimplot:
            left_end, right_end = (left_end, right_end) if trimprct is None else (pl.col(target_col).quantile(trimprct), pl.col(target_col).quantile(1-trimprct))
            fig2 = px.histogram(
                self.df.filter(
                    pl.col(target_col).is_between( left_end, right_end)
                ),
                x = target_col,
                nbins = 50,
            )
        return fig1, fig2


    def show_top_brand(self, brand_col='manufacturer'):
        brand_counts = self.df.select(
            pl.col(brand_col).value_counts()
        ).unnest(
            brand_col
        ).fill_null(
            'Unknown'
        ).sort(
            'count', 
            descending=True
        )

        fig = px.bar(
            brand_counts,
            x = brand_col,
            y = 'count',
            labels={brand_col: 'Brand', 'count': '#Records'}
        )
        return fig

    
    def plot_map(self, lat_col, lon_col):
        fig = px.scatter_map(
            df = self.df,
            x = lon_col,
            y = lat_col,
            center = [40, -120],
            zoom = 3,
            map_style='carto-positron'
        )
        return fig
        

    def show_correlation_matrix(self):
        cols = [col for col in self.df.columns if self.df[col].dtype in (pl.Int64, pl.Float64, pl.Float32)]
        corr_mat = self.df.select(cols).to_pandas().corr(method="spearman")
        fig = px.imshow(
            corr_mat,
            origin='lower',
            color_continuous_scale='rdbu',
            text_auto='.2f',
            title='Correlation Matrix'
            )

        fig.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(
                              showgrid=False,
                              autorange="reversed"
                          ))
        return fig
    

    def show_feature_dist(self, col):
        fig = px.histogram(
            self.df,
            x = col
        )
        return fig
       

    def show_feature_joint_dist(self, cola, colb, target_col=None):
       if target_col is not None:
           fig = px.scatter(
                self.df.sample(fraction=0.5, seed=42),
                x = cola,
                y = colb,
                size=target_col,
                size_max=40,
            )
       else:
           fig = px.density_contour(
                self.df,#.sample(fraction=0.5, seed=42),
                x = cola,
                y = colb,
                marginal_x = 'histogram',
                marginal_y = 'histogram',
                # size=target_col
            )
           
       return fig
    
    def explore_feature_target(self, col, target_col, log_target=False):
        if self.df[col].dtype in (pl.Int64, pl.Float64, pl.Float32):
            fig = px.scatter(
                self.df,
                x = col,
                y = target_col,
                log_y=log_target,
                # marginal_x = 'rug',
                # marginal_y = 'rug',
            )
        else:
            top_k = self.df.group_by(
                col
            ).agg(
                pl.len().alias('count')
            ).top_k(
                10,
                by = 'count'
            )

            fig = px.box(
                self.df.filter(pl.col(col).is_in(top_k[col])).sort(col),
                x = col,
                y = target_col,
                log_y=log_target,
            )
        return fig