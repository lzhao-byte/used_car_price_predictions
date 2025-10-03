from sklearn.preprocessing import (
    PowerTransformer, 
    MinMaxScaler,
    StandardScaler,
    OrdinalEncoder, 
    OneHotEncoder, 
    TargetEncoder
    )
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
import numpy as np
import polars as pl
import time
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    mean_squared_error, 
    root_mean_squared_error, 
    r2_score
)
from sklearn import tree
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## basic random search for hyperparameter tuning
## Bayesian optimization can also be used
## default parameters are used if no tuning is specified
class ModelBuilder():
    input = None
    target = None
    model = None
    encoder = None
    scaler = None
    sim = None
    pipe = None
    data = None
    features = None
    family = None
    name = None


    def __init__(self, df, ylabel='price'):
        self.data = df.with_columns(pl.col(ylabel).cast(pl.Float64)).to_pandas()
        self.target = ylabel


    def _random_forest(self, tune=False, metric='mse'):
        model = RandomForestRegressor(n_estimators=50, max_depth=7, min_samples_leaf=10)
        if tune:
            paras = {
                "n_estimators": range(50, 300, 50),
                "min_samples_split": range(5, 30),
                "min_samples_leaf": range(2, 20),
                "max_depth": range(2, 20),
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True, False]
            }
            model = RandomizedSearchCV(
                estimator = RandomForestRegressor(),
                param_distributions = paras,
                scoring = "neg_mean_squared_error" if metric=='mse' else 'neg_mean_absolute_error',
                n_iter = 30, 
                cv = 5,
                n_jobs = -1,
            )
        return model


    def _xgboost(self, tune=False, metric='mse'):
        model = xgb.XGBRegressor(enable_categorical=True)
        if tune:
            paras = {
                "n_estimators": range(50, 500, 50),
                "learning_rate": np.arange(0.01, 0.1),
                "subsample": np.arange(0.3, 1, 0.1),
                "max_depth": range(2, 20),
            }
            model = RandomizedSearchCV(
                estimator = xgb.XGBRegressor(enable_categorical=True),
                param_distributions = paras,
                scoring = "neg_mean_squared_error" if metric=='mse' else 'neg_mean_absolute_error',
                n_iter = 30, 
                cv = 5,
                n_jobs = -1,
            )
        return model
    
    def _neighbor(self, tune=False, metric='mse'):
        model = KNeighborsRegressor()
        if tune:
            paras = {
                "n_neighbors": range(2, 21),
                "metric": ['euclidean', 'manhattan'],
            }
            model = RandomizedSearchCV(
                estimator = KNeighborsRegressor(),
                param_distributions = paras,
                scoring = "neg_mean_squared_error" if metric=='mse' else 'neg_mean_absolute_error',
                n_iter = 30, 
                cv = 5,
                n_jobs = -1,
            )
        return model

    
    def _mlp(self, tune=False, metric='mse'):
        model = MLPRegressor()
        if tune:
            paras = {
                "hidden_layer_sizes": [(32, ), (64, ), (32, 32), (32, 64, 32)],
                "activation": ['relu', 'tanh'],
                "solver": ['adam', 'lbfgs'],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ['constant', 'adaptive'],
                "max_iter": [200, 500, 1000]
            }
            model = RandomizedSearchCV(
                estimator = MLPRegressor(),
                param_distributions = paras,
                scoring = "neg_mean_squared_error" if metric=='mse' else 'neg_mean_absolute_error',
                n_iter = 30, 
                cv = 5,
                n_jobs = -1,
            )
        return model


    def _linear(self, tune=False, metric='mse'):
        model = ElasticNet(l1_ratio=1, alpha=1)
        if tune:
            paras = {
                "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                "tol": [0.0001, 0.001, 0.01],
                "l1_ratio": np.arange(0, 1.1, 0.1),
                "selection": ["cyclic", "random"]
            }
            model = RandomizedSearchCV(
                estimator = Lasso(),
                param_distributions = paras,
                scoring = "neg_mean_squared_error" if metric=='mse' else 'neg_mean_absolute_error',
                n_iter = 30, 
                cv = 5,
                n_jobs = -1,
            )
        return model
    
    
    def _tree(self, tune=False, metric='mse'):
        model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10)
        if tune:
            paras = {
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'ent']
            }
            model = RandomizedSearchCV(
                estimator = DecisionTreeRegressor(),
                param_distributions = paras,
                scoring = "neg_mean_squared_error" if metric=='mse' else 'neg_mean_absolute_error',
                n_iter = 30, 
                cv = 5,
                n_jobs = -1,
            )
        return model
    
    
    def _setup_model(self, algo="random_forest", tuning=False):
        self.name = algo

        if algo == "xgboost":
            model = self._xgboost(tune=tuning)
        elif algo == "neural_network":
            model = self._mlp(tune=tuning)
        elif algo == "linear":
            model = self._linear(tune=tuning)
        elif algo == 'decision_tree':
            model = self._tree(tune=tuning)
        elif algo == 'nearest_neighbors':
            model = self._neighbor(tune=tuning)
        else:
            model = self._random_forest(tune=tuning)
        self.model = model

        if algo in ('xgboost', 'decision_tree', 'random_forest'):
            self.family = 'tree'
        elif algo in ('linear'):
            self.family = 'linear'
        elif algo in ('nearest_neighbors'):
            self.family = 'neighbors'
        else:
            self.family = 'ann'
        

    def _setup_encoder(self, train_opt='label'):
        if train_opt == 'ordinal':
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        elif train_opt == "one-hot":
            encoder = OneHotEncoder(handle_unknown="ignore")
        elif train_opt == "target":
            encoder = TargetEncoder(target_type="continuous", smooth=10)
        self.encoder = encoder


    def _setup_scaler(self, method=None):
        self.scaler = MinMaxScaler() if method=='minmax' else StandardScaler()
        

    def _split_data(self, test_size=0.2, random_state=42):
        df = self.data.copy()
        # hold for sim
        simset = df.tail(1000)
        df = df[~df.index.isin(simset.index)]
        traincols = df.columns.difference([self.target])
        targetcol = self.target
        x_train, x_test, y_train, y_test = train_test_split(df[traincols], 
                                                            df[targetcol].to_numpy().ravel(), 
                                                            test_size=test_size,
                                                            random_state=random_state)
        # idx_col = 'idx'
        # df = df.with_row_count(name=idx_col)
        # simset = df.tail(1000)
        # df = df.filter(~pl.col(idx_col).is_in(simset[idx_col]))
        # traincols = set(df.columns) - set([self.target])
        # targetcol = self.target
        # x_train, x_test, y_train, y_test = train_test_split(df.select(traincols), 
        #                                                     df[targetcol].to_numpy().ravel(), 
        #                                                     test_size=test_size,
        #                                                     random_state=42)
        self.sim = simset
        self.input = {
            "x_train": x_train, 
            "x_test": x_test, 
            "y_train": y_train, 
            "y_test": y_test
        }
        return x_train, x_test, y_train, y_test


    def _build_tranform_pipe(self):
        # catcols = self.data.select([pl.selectors.string(), pl.selectors.categorical()]).columns
        # numcols = self.data.select(pl.selectors.numeric()).columns
        catcols = list(self.data.select_dtypes(include='object').columns)
        numcols = list(self.data.select_dtypes(include='number').columns)
        numcols.remove(self.target)
        self.features = catcols + numcols
        transformer = ColumnTransformer(
            [
                ("Encoder", self.encoder, catcols),
                ("Scaler", self.scaler, numcols)
            ]
        )
        pipe = Pipeline(
            [
                ('transformer', transformer),
                ('model', self.model)
            ]
        )
        self.pipe = pipe


    def train(self, 
            sel_model='random_forest', 
            sel_train_opt="label", 
            test_size=0.2,
            enable_tuning=False):
        ## encoding
        yield f"Setting up the feature encoder for categorical variables..."
        self._setup_encoder(train_opt=sel_train_opt)
        time.sleep(0.5)
        yield f"Setting up the scaler before feeding into the model..."
        self._setup_scaler()
        time.sleep(0.5)
        yield f"Split data into train and test set..."
        x_train, x_test, y_train, y_test = self._split_data(test_size=test_size)
        time.sleep(0.5)
        yield "Setting up the model..."
        self._setup_model(algo=sel_model, tuning=enable_tuning)
        self._build_tranform_pipe()
        time.sleep(0.5)
        yield "Training..."
        try:
            self.pipe.fit(x_train, y_train)
            cv_score = cross_val_score(self.pipe,
                                       x_train, y_train,
                                       cv=KFold()
                                       )
            yield f"Cross Validation Score ($R^2$): {cv_score.mean(): .2f}."
            # yield f"Test Score (R^2): {r2_score(y_test, self.pipe.predict(x_test)): .2f}."
        except Exception as e:
            yield f"Error: {e}. Please check."


    def predict(self, data=None, target_col='price'):
        if data is not None:
            y_pred = self.pipe.predict(data[data.columns.difference([target_col])])
        else:
            y_pred = self.pipe.predict(self.input['x_test']) 
        return y_pred
    

    def evaluate(self, y_true=None, y_pred=None):
        y_true = self.input['y_test'] if y_true is None else y_true
        y_pred = self.predict() if y_pred is None else y_pred
        
        mse = mean_squared_error(y_pred=y_pred, y_true=y_true)
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
        mape = mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true)
        rmse = root_mean_squared_error(y_pred=y_pred, y_true=y_true)
        r2 = r2_score(y_pred=y_pred, y_true=y_true)
        return {
            "Mean Absolute Error": f"{mae:,.2f}", 
            "Mean Absolute Percentage Error": f"{mape:,.2f}", 
            "Root Mean Squared Error": f"{rmse:.2f}",
            # "Mean Squared Error": f"{mse:,.2f}", 
            "R Squared": f"{r2:.2f}",
        }
    

    def _show_model(self):
        return self.pipe
    

    def _plot_tree(self):
        tree_r = self.model
        fig, ax = plt.subplots(figsize=(10,4))
        print(self.name)
        if self.name == 'decision_tree':
            tree.plot_tree(tree_r, max_depth=3, 
                            feature_names=self.features, 
                            rounded=True, 
                            filled=True,
                            precision=2,
                            ax=ax)
        elif self.name == 'random_forest':
            tree_r = tree_r.estimators_[0]
            tree.plot_tree(tree_r, max_depth=3, 
                            feature_names=self.features, 
                            rounded=True, 
                            filled=True,
                            precision=2,
                            ax=ax)
        elif self.name == 'xgboost':
            xgb.plot_tree(tree_r, num_tree=1, ax=ax)
        else:
            pass
        return fig
        
    
    def _plot_feature_importance(self):
        fig, ax = plt.subplots(figsize=(10,4))
        imp = permutation_importance(self.pipe, 
                                     self.input['x_train'], 
                                     self.input['y_train'])
        sorted_imp_index = imp.importances_mean.argsort()
        ticklabels = {
            "tick_labels": self.input['x_train'].columns[sorted_imp_index]
        }
        ax.boxplot(imp.importances[sorted_imp_index].T, vert=False, **ticklabels)
        ax.axvline(x=0, color="k", linestyle="--")
        return fig
        
    def plot_pred_vs_true(self, y_true, y_pred, title=None):
        fig = make_subplots(1, 2)

        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name='Observations vs Predictions'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(
                x = y_pred-y_true,
                name='Prediction Errors'
            ),
            row=1, col=2
        )
        fig.update_layout(xaxis=dict(title='Observations'), 
                          yaxis=dict(title='Predictions'),
                          xaxis2=dict(title='Prediction Errors'),
                          title=title)
        return fig

    def _show_structure(self):
        feature_names = self.pipe.named_steps['transformer'].get_feature_names_out()
        right_hand_side = "+".join([f"{coef:.0f} \cdot x_{{{fea.replace('_',' ').lower()}}}" for fea, coef in zip(feature_names, self.model.coef_)])
        model_str = f"""Linear Model Equation:\n\n $y={right_hand_side}$."""
        return model_str
    

    def plot_data_dist(self, y_train, y_new, title=None):
        fig = make_subplots(1, 2)

        fig.add_trace(
            go.Histogram(
                x=y_train,
                name='Train Data',
                nbinsx=50,
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(
                x = y_new,
                name='New Data',
                nbinsx=50,
            ),
            row=1, col=2
        )
        fig.update_layout(title=title)
        return fig
    

    def get_data_summary(self, dt):
        return {
            "Average": np.mean(dt),
            "Standard Deviation": np.std(dt),
            "Max Value": np.max(dt),
            "Min Value": np.min(dt),
            "Median": np.median(dt)

        }

