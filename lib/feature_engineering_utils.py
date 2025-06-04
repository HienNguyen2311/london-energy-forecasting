import warnings
from typing import List, Tuple
import time
import humanize
import pandas as pd
from pandas.api.types import is_list_like
from window_ops.rolling import (
    seasonal_rolling_max,
    seasonal_rolling_mean,
    seasonal_rolling_min,
    seasonal_rolling_std,
)
import re
import numpy as np
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

class LogTime:
    def __init__(self, verbose=True, **humanize_kwargs) -> None:
        if "minimum_unit" not in humanize_kwargs.keys():
            humanize_kwargs["minimum_unit"] = 'microseconds'
        self.humanize_kwargs = humanize_kwargs
        self.elapsed = None
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """
        Exceptions are captured in *args, we’ll handle none, since failing can be timed anyway
        """
        self.elapsed = time.time() - self.start
        self.elapsed_str = humanize.precisedelta(self.elapsed, **self.humanize_kwargs)
        if self.verbose:
            print(f"Time Elapsed: {self.elapsed_str}")

def _get_32_bit_dtype(x):
    dtype = x.dtype
    if dtype.name.startswith("float"):
        redn_dtype = "float32"
    elif dtype.name.startswith("int"):
        redn_dtype = "int32"
    else:
        redn_dtype = None
    return redn_dtype

ALLOWED_AGG_FUNCS = ["mean", "max", "min", "std"]
SEASONAL_ROLLING_MAP = {
    "mean": seasonal_rolling_mean,
    "min": seasonal_rolling_min,
    "max": seasonal_rolling_max,
    "std": seasonal_rolling_std,
}


"""
Different ways of creating lags and runtimes
1.
train_df['lag1']=train_df.groupby(["LCLid"])['energy_consumption'].shift(1)
723 ms ± 13.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
2.
train_df['lag1']=train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: x.shift(1))
1.63 s ± 50.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
3.
from window_ops.shift import shift_array
train_df['lag1'] = train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: shift_array(x.values, 1))
1.58 s ± 27.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

"""


def add_lags(
    df: pd.DataFrame,
    lags: List[int],
    column: str,
    ts_id: str = None,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Create Lags for the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        lags (List[int]): List of lags to be created
        column (str): Name of the column to be lagged
        ts_id (str, optional): Column name of Unique ID of a time series to be grouped by before applying the lags.
            If None assumes dataframe only has a single timeseries. Defaults to None.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """
    assert is_list_like(lags), "`lags` should be a list of all required lags"
    assert (
        column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    if ts_id is None:
        warnings.warn(
            "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
        )
        # Assuming just one unique time series in dataset
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_lag_{l}": df[column].shift(l).astype(_32_bit_dtype)
                for l in lags
            }
        else:
            col_dict = {f"{column}_lag_{l}": df[column].shift(l) for l in lags}
    else:
        assert (
            ts_id in df.columns
        ), "`ts_id` should be a valid column in the provided dataframe"
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_lag_{l}": df.groupby([ts_id])[column]
                .shift(l)
                .astype(_32_bit_dtype)
                for l in lags
            }
        else:
            col_dict = {
                f"{column}_lag_{l}": df.groupby([ts_id])[column].shift(l) for l in lags
            }
    df = df.assign(**col_dict)
    added_features = list(col_dict.keys())
    return df, added_features


"""
Different ways of calculating rolling statistics
1.
train_df["rolling_3_mean"]=train_df.groupby(["LCLid"])['energy_consumption'].shift(1).rolling(3).mean()
1.02 s ± 11.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
2.
train_df["rolling_3_mean"]=train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: x.shift(1).rolling(3).mean())
1.92 s ± 45.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
3.
train_df["rolling_3_mean"]=train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: rolling_mean(x.shift(1).values, window_size=3))
1.67 s ± 17.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
4. (for multiple aggregations)
train_df.groupby(["LCLid"])['energy_consumption'].shift(1).rolling(3).agg({"rolling_3_mean": "mean", "rolling_3_std": "std"})
1.8 s ± 26.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""


def time_features_from_frequency_str(freq_str: str) -> List[str]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    """

    features_by_offsets = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.MonthBegin: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.MonthEnd: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.Week: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
        ],
        offsets.Day: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.BusinessDay: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.Hour: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
        ],
        offsets.Minute: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
            "Minute",
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return feature

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:

        Y, YS   - yearly
            alias: A
        M, MS   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
    """
    raise RuntimeError(supported_freq_msg)


def add_temporal_features(
    df: pd.DataFrame,
    field_name: str,
    frequency: str,
    add_elapsed: bool = True,
    prefix: str = None,
    drop: bool = True,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Adds columns relevant to a date in the column `field_name` of `df`.

    Args:
        df (pd.DataFrame): Dataframe to which the features need to be added
        field_name (str): The date column which should be encoded using temporal features
        frequency (str): The frequency of the date column so that only relevant features are added.
            If frequency is "Weekly", then temporal features like hour, minutes, etc. doesn't make sense.
        add_elapsed (bool, optional): Add time elapsed as a monotonically increasing function. Defaults to True.
        prefix (str, optional): Prefix to the newly created columns. If left None, will use the field name. Defaults to None.
        drop (bool, optional): Flag to drop the data column after feature creation. Defaults to True.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """
    field = df[field_name]
    prefix = (re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix) + "_"
    attr = time_features_from_frequency_str(frequency)
    _32_bit_dtype = "int32"
    added_features = []
    for n in attr:
        if n == "Week":
            continue
        df[prefix + n] = (
            getattr(field.dt, n.lower()).astype(_32_bit_dtype)
            if use_32_bit
            else getattr(field.dt, n.lower())
        )
        added_features.append(prefix + n)
    # Pandas removed `dt.week` in v1.1.10
    if "Week" in attr:
        week = (
            field.dt.isocalendar().week
            if hasattr(field.dt, "isocalendar")
            else field.dt.week
        )
        df.insert(
            3, prefix + "Week", week.astype(_32_bit_dtype) if use_32_bit else week
        )
        added_features.append(prefix + "Week")
    if add_elapsed:
        mask = ~field.isna()
        df[prefix + "Elapsed"] = np.where(
            mask, field.values.astype(np.int64) // 10**9, None
        )
        if use_32_bit:
            if df[prefix + "Elapsed"].isnull().sum() == 0:
                df[prefix + "Elapsed"] = df[prefix + "Elapsed"].astype("int32")
            else:
                df[prefix + "Elapsed"] = df[prefix + "Elapsed"].astype("float32")
        added_features.append(prefix + "Elapsed")
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df, added_features