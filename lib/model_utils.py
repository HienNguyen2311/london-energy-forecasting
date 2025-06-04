import plotly.graph_objects as go
from itertools import cycle
import pandas as pd
import time
from statsforecast.core import StatsForecast
from statsforecast.models import Naive
from typing import Callable, Union, List, Dict
import numpy as np
import plotly.express as px
from dataclasses import MISSING, field, dataclass
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
import copy
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_squared_error as mse
from darts.metrics import mse as d_mse
from darts.metrics import rmse as d_rmse
from darts.metrics import mae as d_mae
import warnings
import torch

def intersect_list(list1, list2):
    return list(set(list1).intersection(set(list2)))

def difference_list(list1, list2):
    return list(set(list1) - set(list2))

def plot_forecast(pred_df, forecast_columns, timestamp_col, forecast_display_names=None,
                  target_col_name='energy_consumption_imputed'):
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)

    mask = ~pred_df[forecast_columns[0]].isnull()
    colors = [c.replace("rgb", "rgba").replace(")", ", <alpha>)") for c in px.colors.qualitative.Dark2]
    act_color = colors[0]
    colors = cycle(colors[1:])
    dash_types = cycle(["dash", "dot", "dashdot"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df[mask][timestamp_col], y=pred_df[mask][target_col_name],
                             mode='lines', line=dict(color=act_color.replace("<alpha>", "0.3")),
                             name='Actual Consumption'))

    for col, display_col in zip(forecast_columns, forecast_display_names):
        fig.add_trace(go.Scatter(x=pred_df[mask][timestamp_col], y=pred_df.loc[mask, col],
                                 mode='lines', line=dict(dash=next(dash_types), color=next(colors).replace("<alpha>", "1")),
                                 name=display_col))
    return fig

def evaluate_performance(ts_train, ts_test, models, metrics, freq, level, id_col, time_col, target_col, h, metric_df=None):
    if metric_df is None:
        metric_df = pd.DataFrame()  # Initialize an empty DataFrame if not provided

    results = ts_test.copy()

    # Timing dictionary to store train and predict durations
    timing = {}

    for model in models:
        model_name = model.__class__.__name__
        evaluation = {}  # Reset the evaluation dictionary for each model

        # Start the timer for fitting and prediction
        start_time = time.time()

        # Instantiate StatsForecast class
        sf = StatsForecast(
            models=[model],
            freq=freq,
            n_jobs=-1,
            fallback_model=Naive()
        )

        # Efficiently predict without storing memory
        y_pred = sf.forecast(
            h=h,
            df=ts_train,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
        )

        # Calculating the duration
        duration = time.time() - start_time
        timing[model_name] = duration

        # Merge prediction results to the original dataframe
        results = results.merge(y_pred, how='left', on=[id_col, time_col])

        ids = ts_train[id_col].unique()
        # Calculate metrics
        for id in ids:
            temp_results = results[results[id_col] == id]
            for metric in metrics:
                metric_name = metric.__name__
                evaluation[metric_name] = metric(temp_results[target_col].values, temp_results[model_name].values)
            evaluation[id_col] = id
            evaluation['Time Elapsed'] = timing[model_name]

            # Prepare and append this model's results to metric_df
            temp_df = pd.DataFrame(evaluation, index=[0])
            temp_df['Model'] = model_name
            metric_df = pd.concat([metric_df, temp_df], ignore_index=True)

    return results, metric_df

def forecast_bias(actual_series: Union[ np.ndarray],
        pred_series: Union[ np.ndarray],
        intersect: bool = True,
        *,
        reduction: Callable[[np.ndarray], float] = np.mean,
        inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
        n_jobs: int = 1,
        verbose: bool = False) -> Union[float, np.ndarray]:
    """ Forecast Bias (FB).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{\\sum_{t=1}^{T}{y_t}
              - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` of actual values.
    pred_series
        The `TimeSeries` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate TimeSeries instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        TimeSeries. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a TimeSeries is
        passed as input, parallelising operations regarding different TimeSerie`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If :math:`\\sum_{t=1}^{T}{y_t} = 0`.

    Returns
    -------
    float
        The Forecast Bias (OPE)
    """
    assert type(actual_series) is type(pred_series), "actual_series and pred_series should be of same type."
    if isinstance(actual_series, np.ndarray):
        y_true, y_pred = actual_series, pred_series
    else:
        y_true = actual_series
        y_pred = pred_series
    #     y_true, y_pred = _get_values_or_raise(actual_series, pred_series, intersect)
    #y_true, y_pred = _remove_nan_union(y_true, y_pred)
    y_true_sum, y_pred_sum = np.sum(y_true), np.sum(y_pred)
    # raise_if_not(y_true_sum > 0, 'The series of actual value cannot sum to zero when computing OPE.', logger)
    return ((y_true_sum - y_pred_sum) / y_true_sum) * 100

@dataclass
class FeatureConfig:

    date: List = field(
        default=MISSING,
        metadata={"help": "Column name of the date column"},
    )
    target: str = field(
        default=MISSING,
        metadata={"help": "Column name of the target column"},
    )

    original_target: str = field(
        default=None,
        metadata={
            "help": "Column name of the original target column in acse of transformed target. If None, it will be assigned same value as target"
        },
    )

    continuous_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the categorical fields. Defaults to []"},
    )
    boolean_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the boolean fields. Defaults to []"},
    )

    index_cols: str = field(
        default_factory=list,
        metadata={
            "help": "Column names which needs to be set as index in the X and Y dataframes."
        },
    )
    exogenous_features: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names of the exogenous features. Must be a subset of categorical and continuous features"
        },
    )
    feature_list: List[str] = field(init=False)

    def __post_init__(self):
        assert (
            len(self.categorical_features) + len(self.continuous_features) > 0
        ), "There should be at-least one feature defined in categorical or continuous columns"
        self.feature_list = (
            self.categorical_features + self.continuous_features + self.boolean_features
        )
        assert (
            self.target not in self.feature_list
        ), f"`target`({self.target}) should not be present in either categorical, continuous or boolean feature list"
        assert (
            self.date not in self.feature_list
        ), f"`date`({self.target}) should not be present in either categorical, continuous or boolean feature list"
        extra_exog = set(self.exogenous_features) - set(self.feature_list)
        assert (
            len(extra_exog) == 0
        ), f"These exogenous features are not present in feature list: {extra_exog}"
        intersection = (
            set(self.continuous_features)
            .intersection(self.categorical_features + self.boolean_features)
            .union(
                set(self.categorical_features).intersection(
                    self.continuous_features + self.boolean_features
                )
            )
            .union(
                set(self.boolean_features).intersection(
                    self.continuous_features + self.categorical_features
                )
            )
        )
        assert (
            len(intersection) == 0
        ), f"There should not be any overlaps between the categorical contonuous and boolean features. {intersection} are present in more than one definition"
        if self.original_target is None:
            self.original_target = self.target

        self.encoder = None

    def get_X_y(
        self, df: pd.DataFrame, categorical: bool = False, exogenous: bool = False, test: bool = False
    ):
        feature_list = copy.deepcopy(self.continuous_features)
        if categorical:
            feature_list += self.categorical_features + self.boolean_features
            # Add one-hot encoding
            if not test:  # Training data
                if self.encoder is None:
                    self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

                categorical_data = df[self.categorical_features]
                encoded_categorical = self.encoder.fit_transform(categorical_data)
                encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_features)

            else:  # Test data
                if self.encoder is None:
                    raise ValueError("Encoder not fitted. Call with test=False first.")

                categorical_data = df[self.categorical_features]
                encoded_categorical = self.encoder.transform(categorical_data)
                encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_features)

            # Add encoded features to dataframe
            encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)

            # Update feature list
            feature_list = self.continuous_features + self.boolean_features + list(encoded_feature_names)

        if not exogenous:
            feature_list = list(set(feature_list) - set(self.exogenous_features))
        feature_list = list(set(feature_list))
        delete_index_cols = list(set(self.index_cols) - set(self.feature_list))
        (X, y, y_orig) = (
            #df.loc[:, set(feature_list + self.index_cols)]
            df.loc[:, list(set(feature_list + self.index_cols))]
            .set_index(self.index_cols, drop=False)
            .drop(columns=delete_index_cols),
            df.loc[:, [self.target] + self.index_cols].set_index(
                self.index_cols, drop=True
            )
            if self.target in df.columns
            else None,
            df.loc[:, [self.original_target] + self.index_cols].set_index(
                self.index_cols, drop=True
            )
            if self.original_target in df.columns
            else None,
        )
        return X, y, y_orig


@dataclass
class ModelConfig:

    model: BaseEstimator = field(
        default=MISSING, metadata={"help": "Sci-kit Learn Compatible model instance"}
    )

    name: str = field(
        default=None,
        metadata={
            "help": "Name or identifier for the model. If left None, will use the string representation of the model"
        },
    )

    normalize: bool = field(
        default=False,
        metadata={"help": "Flag whether to normalize the input or not"},
    )
    fill_missing: bool = field(
        default=True,
        metadata={"help": "Flag whether to fill missing values before fitting"},
    )
    encode_categorical: bool = field(
        default=False,
        metadata={"help": "Flag whether to encode categorical values before fitting"},
    )
    categorical_encoder: BaseEstimator = field(
        default=None,
        metadata={"help": "Categorical Encoder to be used"},
    )

    def __post_init__(self):
        assert not (
            self.encode_categorical and self.categorical_encoder is None
        ), "`categorical_encoder` cannot be None if `encode_categorical` is True"

    def clone(self):
        self.model = clone(self.model)
        return self


@dataclass
class MissingValueConfig:

    bfill_columns: List = field(
        default_factory=list,
        metadata={"help": "Column names which should be filled using strategy=`bfill`"},
    )

    ffill_columns: List = field(
        default_factory=list,
        metadata={"help": "Column names which should be filled using strategy=`ffill`"},
    )
    zero_fill_columns: List = field(
        default_factory=list,
        metadata={"help": "Column names which should be filled using 0"},
    )

    def impute_missing_values(self, df: pd.DataFrame):
        df = df.copy()
        bfill_columns = intersect_list(df.columns, self.bfill_columns)
        df[bfill_columns] = df[bfill_columns].fillna(method="bfill")
        ffill_columns = intersect_list(df.columns, self.ffill_columns)
        df[ffill_columns] = df[ffill_columns].fillna(method="ffill")
        zero_fill_columns = intersect_list(df.columns, self.zero_fill_columns)
        df[zero_fill_columns] = df[zero_fill_columns].fillna(0)
        check = df.isnull().any()
        missing_cols = check[check].index.tolist()
        missing_numeric_cols = intersect_list(
            missing_cols, df.select_dtypes([np.number]).columns.tolist()
        )
        missing_object_cols = intersect_list(
            missing_cols, df.select_dtypes(["object"]).columns.tolist()
        )
        # Filling with mean and NA as default fillna strategy
        df[missing_numeric_cols] = df[missing_numeric_cols].fillna(
            df[missing_numeric_cols].mean()
        )
        df[missing_object_cols] = df[missing_object_cols].fillna("NA")
        return df

class MLForecast:
    def __init__(
        self,
        model_config: ModelConfig,
        feature_config: FeatureConfig,
        missing_config: MissingValueConfig = None,
        target_transformer: object = None,
    ) -> None:
        """Convenient wrapper around scikit-learn style estimators

        Args:
            model_config (ModelConfig): Instance of the ModelConfig object defining the model
            feature_config (FeatureConfig): Instance of the FeatureConfig object defining the features
            missing_config (MissingValueConfig, optional): Instance of the MissingValueConfig object
                defining how to fill missing values. Defaults to None.
            target_transformer (object, optional): Instance of target transformers from src.transforms.
                Should support `fit`, `transform`, and `inverse_transform`. It should also
                return `pd.Series` with datetime index to work without an error. Defaults to None.
        """
        self.model_config = model_config
        self.feature_config = feature_config
        self.missing_config = missing_config
        self.target_transformer = target_transformer
        self._model = clone(model_config.model)
        if self.model_config.normalize:
            self._scaler = StandardScaler()
        if self.model_config.encode_categorical:
            self._cat_encoder = self.model_config.categorical_encoder
            self._encoded_categorical_features = copy.deepcopy(
                self.feature_config.categorical_features
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        is_transformed: bool = False,
        fit_kwargs: Dict = {},
    ):
        """Handles standardization, missing value handling, and training the model

        Args:
            X (pd.DataFrame): The dataframe with the features as columns
            y (Union[pd.Series, np.ndarray]): Dataframe, Series, or np.ndarray with the targets
            is_transformed (bool, optional): Whether the target is already transformed.
            If `True`, fit wont be transforming the target using the target_transformer
                if provided. Defaults to False.
            fit_kwargs (Dict, optional): The dictionary with keyword args to be passed to the
                fit funciton of the model. Defaults to {}.
        """
        missing_feats = difference_list(X.columns, self.feature_config.feature_list)
        if len(missing_feats) > 0:
            warnings.warn(
                f"Some features in defined in FeatureConfig is not present in the dataframe. Ignoring these features: {missing_feats}"
            )
        self._continuous_feats = intersect_list(
            self.feature_config.continuous_features, X.columns
        )
        self._categorical_feats = intersect_list(
            self.feature_config.categorical_features, X.columns
        )
        self._boolean_feats = intersect_list(
            self.feature_config.boolean_features, X.columns
        )
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)

        # if self.model_config.encode_categorical:
        #     missing_cat_cols = difference_list(
        #         self._categorical_feats,
        #         self.model_config.categorical_encoder.cols,
        #     )
        #     assert (
        #         len(missing_cat_cols) == 0
        #     ), f"These categorical features are not handled by the categorical_encoder : {missing_cat_cols}"
        #     # In later versions of sklearn get_feature_names have been deprecated
        #     try:
        #         feature_names = self.model_config.categorical_encoder.get_feature_names()
        #     except AttributeError:
        #         # in favour of get_feature_names_out()
        #         feature_names = self.model_config.categorical_encoder.get_feature_names_out()
        #     X = self._cat_encoder.fit_transform(X, y)
        #     self._encoded_categorical_features = difference_list(
        #         feature_names,
        #         self.feature_config.continuous_features
        #         + self.feature_config.boolean_features,
        #     )
        # else:
        #     self._encoded_categorical_features = []

        #Fixed Chapt 10 issue to move fit_transform before get_feature_names()
        if self.model_config.encode_categorical:
            missing_cat_cols = difference_list(
                self._categorical_feats,
                self._cat_encoder.cols,
            )
            assert (
                len(missing_cat_cols) == 0
            ), f"These categorical features are not handled by the categorical_encoder: {missing_cat_cols}"

            # Fit the encoder before getting feature names
            X = self._cat_encoder.fit_transform(X, y)

            # Now get the feature names from the fitted encoder
            try:
                feature_names = self._cat_encoder.get_feature_names()
            except AttributeError:
                # For newer versions of sklearn
                feature_names = self._cat_encoder.get_feature_names_out()

            self._encoded_categorical_features = difference_list(
                feature_names,
                self.feature_config.continuous_features + self.feature_config.boolean_features,
            )
        else:
            self._encoded_categorical_features = []



        if self.model_config.normalize:
            X[
                self._continuous_feats + self._encoded_categorical_features
            ] = self._scaler.fit_transform(
                X[self._continuous_feats + self._encoded_categorical_features]
            )
        self._train_features = X.columns.tolist()
        # print(len(self._train_features))
        if not is_transformed and self.target_transformer is not None:
            y = self.target_transformer.fit_transform(y)
        self._model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predicts on the given dataframe using the trained model

        Args:
            X (pd.DataFrame): The dataframe with the features as columns. The index is passed on to the prediction series

        Returns:
            pd.Series: predictions using the model as a pandas Series with datetime index
        """
        assert len(intersect_list(self._train_features, X.columns)) == len(
            self._train_features
        ), f"All the features during training is not available while predicting: {difference_list(self._train_features, X.columns)}"
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)
        if self.model_config.encode_categorical:
            X = self._cat_encoder.transform(X)
        if self.model_config.normalize:
            X[
                self._continuous_feats + self._encoded_categorical_features
            ] = self._scaler.transform(
                X[self._continuous_feats + self._encoded_categorical_features]
            )
        y_pred = pd.Series(
            self._model.predict(X).ravel(),
            index=X.index,
            name=f"{self.model_config.name}",
        )
        if self.target_transformer is not None:
            y_pred = self.target_transformer.inverse_transform(y_pred)
            y_pred.name = f"{self.model_config.name}"
        return y_pred

    def feature_importance(self) -> pd.DataFrame:
        """Generates the feature importance dataframe, if available. For linear
            models the coefficients are used and tree based models use the inbuilt
            feature importance. For the rest of the models, it returns an empty dataframe.

        Returns:
            pd.DataFrame: Feature Importance dataframe, sorted in descending order of its importances.
        """
        if hasattr(self._model, "coef_") or hasattr(
            self._model, "feature_importances_"
        ):
            feat_df = pd.DataFrame(
                {
                    "feature": self._train_features,
                    "importance": self._model.coef_.ravel()
                    if hasattr(self._model, "coef_")
                    else self._model.feature_importances_.ravel(),
                }
            )
            feat_df["_abs_imp"] = np.abs(feat_df.importance)
            feat_df = feat_df.sort_values("_abs_imp", ascending=False).drop(
                columns="_abs_imp"
            )
        else:
            feat_df = pd.DataFrame()
        return feat_df

def calculate_metrics(
    y: pd.Series, y_pred: pd.Series, name: str, y_train: pd.Series = None
):
    """Method to calculate the metrics given the actual and predicted series

    Args:
        y (pd.Series): Actual target with datetime index
        y_pred (pd.Series): Predictions with datetime index
        name (str): Name or identification for the model
        y_train (pd.Series, optional): Actual train target to calculate MASE with datetime index. Defaults to None.

    Returns:
        Dict: Dictionary with MAE, MSE, MASE, and Forecast Bias
    """
    return {
        "Algorithm": name,
        "MAE": mae(y, y_pred),
        "MSE": mse(y, y_pred),
        "RMSE": rmse(y, y_pred),
        "Forecast Bias": forecast_bias(y, y_pred)
    }

def evaluate_model(
    model_config,
    feature_config,
    missing_config,
    train_features,
    train_target,
    test_features,
    test_target,
):
    ml_model = MLForecast(
        model_config,
        feature_config,
        missing_config,
    )
    ml_model.fit(train_features, train_target)
    y_pred = ml_model.predict(test_features)
    y_pred = pd.Series(
    data=ml_model.predict(test_features).squeeze(),
    index=test_target.index,
    name='predictions'
    )
    test_target = pd.Series(
    data=test_target.squeeze(),  # Removes singleton dimension
    index=test_target.index,
    name='energy_consumption'
    )
    feat_df = ml_model.feature_importance()
    metrics = calculate_metrics(test_target, y_pred, model_config.name, train_target)
    return y_pred, metrics, feat_df

def preprocess_features(df, continuous_features, categorical_features, boolean_features, n_hashed_features,
                        hashed_features_list, bfill_columns=None, ffill_columns=None, zero_fill_columns=None):
    """Preprocess features according to their types."""
    df = df.copy()

    # Step 1: Handle missing values
    print("Handling missing values...")
    # Backward fill specified columns
    if bfill_columns:
        bfill_cols = intersect_list(df.columns, bfill_columns)
        df[bfill_cols] = df[bfill_cols].fillna(method="bfill")

    # Forward fill specified columns
    if ffill_columns:
        ffill_cols = intersect_list(df.columns, ffill_columns)
        df[ffill_cols] = df[ffill_cols].fillna(method="ffill")

    # Zero fill specified columns
    if zero_fill_columns:
        zero_cols = intersect_list(df.columns, zero_fill_columns)
        df[zero_cols] = df[zero_cols].fillna(0)

    # Fill remaining numeric columns with mean
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    remaining_numeric = [col for col in numeric_cols if df[col].isna().any()]
    for col in remaining_numeric:
        df[col] = df[col].fillna(df[col].mean())

    # Fill remaining categorical columns with "NA"
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    remaining_cat = [col for col in cat_cols if df[col].isna().any()]
    for col in remaining_cat:
        df[col] = df[col].fillna("NA")

    # Step 2: Encode categorical features
    print("Encoding categorical features...")
    for col in categorical_features:
        # Create hasher for this column
        hasher = FeatureHasher(n_features=n_hashed_features, input_type='string')
        # Transform column values into required format (list of lists with single string)
        col_data = [[str(val)] for val in df[col].values]
        # Hash the column
        hashed_col = hasher.transform(col_data)
        # Add to list
        hashed_features_list.append(hashed_col)
    # Horizontally stack all hashed features
    all_hashed_features = hstack(hashed_features_list)

    # Convert to DataFrame if needed
    all_hashed_features_df = pd.DataFrame.sparse.from_spmatrix(
        all_hashed_features,
        index=df.index,  # Critical for alignment
        columns=[f'hash_{i}' for i in range(all_hashed_features.shape[1])]
    )
    # 1. Drop original categorical columns
    df_without_cats = df.drop(columns=categorical_features, errors='ignore', axis=1)

    # 3. Combine with original DataFrame (now without categoricals)
    hashed_df = pd.concat([df_without_cats, all_hashed_features_df], axis=1)

    # Step 3: Convert boolean features to integers
    bool_cols_present = intersect_list(boolean_features, hashed_df.columns)
    if bool_cols_present:
        hashed_df[bool_cols_present] = hashed_df[bool_cols_present].astype(int)

    # Step 4: Normalize continuous features
    print("Normalizing features...")
    cont_cols_present = intersect_list(continuous_features, hashed_df.columns)
    if cont_cols_present:
        scaler = StandardScaler()
        hashed_df[cont_cols_present] = scaler.fit_transform(hashed_df[cont_cols_present])

    return hashed_df

def timeseries_to_df(ts, value_col):
    """Convert Darts TimeSeries to pandas DataFrame"""
    return pd.DataFrame({
        'timestamp': ts.time_index,
        value_col: ts.values().flatten()
    })

def ts_forecast_bias(actual, predicted):
    """Calculate forecast bias"""
    # Convert TimeSeries to numpy arrays first
    actual_values = actual.values().flatten()
    predicted_values = predicted.values().flatten()

    # Calculate bias as percentage
    actual_sum = np.sum(actual_values)
    predicted_sum = np.sum(predicted_values)

    return ((actual_sum - predicted_sum) / actual_sum) * 100

def train_and_evaluate_model(model_config, train_series_list, train_covariates_list,
                           test_series_list, test_covariates_list, household_ids,
                           target_scaler, output_dir_pred, output_dir_metric):
    """
    Train and evaluate a single model

    Args:
        model_config: Dictionary containing model configuration
        train_series_list: List of training time series
        train_covariates_list: List of training covariates
        test_series_list: List of test time series
        test_covariates_list: List of test covariates
        household_ids: List of household IDs
        target_scaler: Fitted scaler for inverse transformation
        output_dir: Directory to save results

    Returns:
        predictions_df: DataFrame with predictions
        metrics_df: DataFrame with metrics
    """

    print(f"Training {model_config['name']} model...")

    # Initialize model
    model = model_config['class'](**model_config['params'])

    # Train
    model.fit(
        series=train_series_list,
        future_covariates=train_covariates_list,
        verbose=True
    )

    # Evaluate
    model_predictions = []
    model_metrics = []

    for i, (test_series, test_cov) in enumerate(zip(test_series_list, test_covariates_list)):
        forecast = model.predict(
            n=len(test_series),
            series=train_series_list[i],
            future_covariates=test_cov
        )

        forecast_actual = target_scaler.inverse_transform(forecast)
        test_actual = target_scaler.inverse_transform(test_series)

        # Create prediction DataFrame
        test_df = timeseries_to_df(test_actual, 'actual')
        forecast_df = timeseries_to_df(forecast_actual, 'predicted')

        merged_df = pd.merge(test_df, forecast_df, on='timestamp')
        merged_df['household_id'] = household_ids[i]
        merged_df['model'] = model_config['name']
        model_predictions.append(merged_df)

        # Calculate metrics
        model_metrics.append({
            'household_id': household_ids[i],
            'MSE': d_mse(test_actual, forecast_actual),
            'RMSE': d_rmse(test_actual, forecast_actual),
            'MAE': d_mae(test_actual, forecast_actual),
            'Forecast Bias': ts_forecast_bias(test_actual, forecast_actual),
            'algorithm': model_config['name']
        })

    # Create DataFrames
    predictions_df = pd.concat(model_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(model_metrics)

    # Cleanup to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Save Results
    print("\nSaving final results...")

    predictions_df.to_pickle(output_dir_pred)
    metrics_df.to_pickle(output_dir_metric)

    return predictions_df, metrics_df