
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
from itertools import cycle
from plotly.colors import n_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import pacf, acf
import warnings

def preprocess_compact(x, max_date):
    start_date = x['day'].min()
    name = x['LCLid'].unique()[0]
    ### Fill missing dates with NaN ###
    # Create a date range from  min to max
    dr = pd.date_range(start=x['day'].min(), end=max_date, freq="1D")
    # Add hh_0 to hh_47 to columns and with some unstack magic recreating date-hh_x combinations
    dr = pd.DataFrame(columns=[f"hh_{i}" for i in range(48)], index=dr).unstack().reset_index()
    # renaming the columns
    dr.columns = ["hour_block", "day", "_"]
    # left merging the dataframe to the standard dataframe
    # now the missing values will be left as NaN
    dr = dr.merge(x, on=['hour_block','day'], how='left')
    # sorting the rows
    dr.sort_values(['day',"offset"], inplace=True)
    # extracting the timeseries array
    ts = dr['energy_consumption'].values
    len_ts = len(ts)
    return start_date, name, ts, len_ts

def load_process_block_compact(block_df, max_date, freq="30min", ts_identifier="series_name", value_name="series_value"):
    grps = block_df.groupby('LCLid')
    all_series = []
    all_start_dates = []
    all_names = []
    all_data = {}
    all_len = []
    for idx, df in tqdm(grps, leave=False):
        start_date, name, ts, len_ts = preprocess_compact(df, max_date)
        all_series.append(ts)
        all_start_dates.append(start_date)
        all_names.append(name)
        all_len.append(len_ts)

    all_data[ts_identifier] = all_names
    all_data['start_timestamp'] = all_start_dates
    all_data['frequency'] = freq
    all_data[value_name] = all_series
    all_data['series_length'] = all_len
    return pd.DataFrame(all_data)

def map_weather_holidays(row, bank_holidays, weather_hourly):
    date_range = pd.date_range(row['start_timestamp'], periods=row['series_length'], freq=row['frequency'])
    std_df = pd.DataFrame(index=date_range)
    #Filling Na with NO_HOLIDAY cause rows before earliers holiday will be NaN
    holidays = std_df.join(bank_holidays, how="left").fillna("NO_HOLIDAY")
    weather = std_df.join(weather_hourly, how='left')
    assert len(holidays)==row['series_length'], "Length of holidays should be same as series length"
    assert len(weather)==row['series_length'], "Length of weather should be same as series length"
    row['holidays'] = holidays['Type'].values
    for col in weather:
        row[col] = weather[col].values
    return row

################################################################################

def compact_to_expanded(
    df, timeseries_col, static_cols, time_varying_cols, ts_identifier
):
    def preprocess_expanded(x):
        ### Fill missing dates with NaN ###
        # Create a date range from  start
        dr = pd.date_range(
            start=x["start_timestamp"],
            periods=len(x["energy_consumption"]),
            freq=x["frequency"],
        )
        df_columns = defaultdict(list)
        df_columns["timestamp"] = dr
        for col in [ts_identifier, timeseries_col] + static_cols + time_varying_cols:
            df_columns[col] = x[col]
        return pd.DataFrame(df_columns)

    all_series = []
    for i in tqdm(range(len(df))):
        all_series.append(preprocess_expanded(df.iloc[i]))
    df = pd.concat(all_series)
    del all_series
    return df

def format_plot(fig, legends=None, xlabel="Day", ylabel="Value", figsize=(500, 900), font_size=15, title_font_size=20):
    """
    Format a plotly figure with consistent styling.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to format
    legends : list, optional
        List of legend names to apply to traces
    xlabel : str, default="Day"
        Label for x-axis
    ylabel : str, default="Value"
        Label for y-axis
    figsize : tuple, default=(500, 900)
        Figure size as (height, width)
    font_size : int, default=15
        Font size for legends, axis labels and ticks
    title_font_size : int, default=20
        Font size for the title

    Returns:
    --------
    plotly.graph_objects.Figure
        The formatted figure
    """
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_layout(
        autosize=False,
        width=figsize[1],
        height=figsize[0],
        title={
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                "size": title_font_size
            }
        },
        legend_title=None,
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title_text=ylabel,
            title_font=dict(size=font_size),  # Using title_font instead of titlefont
            tickfont=dict(size=font_size),
        ),
        xaxis=dict(
            title_text=xlabel,
            title_font=dict(size=font_size),  # Using title_font instead of titlefont
            tickfont=dict(size=font_size),
        )
    )
    return fig

def imputation(df):
    #Create a column with the weekday and hour from timestamp
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    #Calculate weekday-hourly average consumption
    day_hourly_profile = df.groupby(['weekday','hour'])['energy_consumption'].mean().reset_index()
    day_hourly_profile.rename(columns={"energy_consumption": "day_hourly_profile"}, inplace=True)
    #Saving the index because it gets lost in merge
    idx = df.index
    #Merge the day-hourly profile dataframe to ts dataframe
    df = df.merge(day_hourly_profile, on=['weekday', 'hour'], how='left', validate="many_to_one")
    df.index = idx
    #Using the day-hourly profile to fill missing
    df['energy_consumption_imputed'] = df['energy_consumption'].fillna(df['day_hourly_profile'])

    return df

def make_lines_greyscale(fig):
    colors = cycle(
        list(
            set(
                n_colors(
                    "rgb(100, 100, 100)", "rgb(200, 200, 200)", 2 + 1, colortype="rgb"
                )
            )
        )
    )
    for d in fig.data:
        d.line.color = next(colors)
    return fig

def two_line_plot_secondary_axis(
    x,
    y1,
    y2,
    y1_name="y1",
    y2_name="y2",
    title="",
    legends=None,
    xlabel="Time",
    ylabel="Value",
    greyscale=False,
    dash_secondary=False,
):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=x, y=y1, name=y1_name),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            name=y2_name,
            line=dict(dash="dash") if dash_secondary else None,
        ),
        secondary_y=True,
    )
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
            autosize=False,
            width=900,
            height=500,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
            title_text=title,
        title_font={"size": 20},
        legend_title=None,
            yaxis=dict(
                title_text=ylabel,
                title_font=dict(size=12),
            ),
            xaxis=dict(
                title_text=xlabel,
                title_font=dict(size=12),
        ),
    )
    if greyscale:
        fig = make_lines_greyscale(fig)
    return fig

def update_figure_layout(fig, font_size=15):
    fig.update_layout(
    legend=dict(
        font=dict(size=font_size),
        orientation="h",
        yanchor="bottom",
        y=0.98,
        xanchor="right",
        x=1,
    ),
    yaxis=dict(
        title_font=dict(size=font_size),
        tickfont=dict(size=font_size),
    ),
    xaxis=dict(
        title_font=dict(size=font_size),
        tickfont=dict(size=font_size),
    )
)
    return fig

def decomposition_plot(
        ts_index, observed=None, seasonal=None, trend=None, resid=None
    ):
        """Plots the decomposition output
        """
        series = []
        if observed is not None:
            series += ["Original"]
        if trend is not None:
            series += ["Trend"]
        if seasonal is not None:
            series += ["Seasonal"]
        if resid is not None:
            series += ["Residual"]
        if len(series) == 0:
            raise ValueError(
                "All component flags were off. Need atleast one of the flags turned on to plot."
            )
        fig = make_subplots(
            rows=len(series), cols=1, shared_xaxes=True, subplot_titles=series
        )
        x = ts_index
        row = 1
        if observed is not None:
            fig.append_trace(
                go.Scatter(x=x, y=observed, name="Original"), row=row, col=1
            )
            row += 1
        if trend is not None:
            fig.append_trace(
                go.Scatter(x=x, y=trend, name="Trend"), row=row, col=1
            )
            row += 1
        if seasonal is not None:
            fig.append_trace(
                go.Scatter(x=x, y=seasonal, name="Seasonal"),
                row=row,
                col=1,
            )
            row += 1
        if resid is not None:
            fig.append_trace(
                go.Scatter(x=x, y=resid, name="Residual"), row=row, col=1
            )
            row += 1

        fig.update_layout(
            title_text="Seasonal Decomposition",
            autosize=False,
            width=1200,
            height=700,
            title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
            title_font={"size": 20},
            legend_title=None,
            showlegend=False,
            legend=dict(
                font=dict(size=15),
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
            ),
            yaxis=dict(
                # title_text=ylabel,
                title_font=dict(size=15),
                tickfont=dict(size=15),
            ),
            xaxis=dict(
                # title_text=xlabel,
                title_font=dict(size=15),
                tickfont=dict(size=15),
            )
        )
        return fig

def plot_autocorrelation(series,vertical=False, figsize=(500, 900), **kwargs):
    if "qstat" in kwargs.keys():
        warnings.warn("`qstat` for acf is ignored as it has no impact on the plots")
        kwargs.pop("qstat")
    acf_args = ["adjusted","nlags", "fft", "alpha", "missing"]
    pacf_args = ["nlags","method","alpha"]
    if "nlags" not in kwargs.keys():
        nobs = len(series)
        kwargs['nlags'] = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
    kwargs['fft'] = True
    acf_kwargs = {k:v for k,v in kwargs.items() if k in acf_args}
    pacf_kwargs = {k:v for k,v in kwargs.items() if k in pacf_args}
    acf_array = acf(series, **acf_kwargs)
    pacf_array = pacf(series, **pacf_kwargs)
    if "alpha" in kwargs.keys():
        acf_array, _ = acf_array
        pacf_array, _ = pacf_array
    x_ = np.arange(1,len(acf_array))
    rows, columns = (2, 1) if vertical else (1,2)
    fig = make_subplots(
            rows=rows, cols=columns, shared_xaxes=True, shared_yaxes=False, subplot_titles=['Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)']
        )
    #ACF
    row, column = 1, 1
    [fig.append_trace(go.Scatter(x=(x,x), y=(0,acf_array[x]), mode='lines',line_color='#3f3f3f'), row=row, col=column) 
     for x in range(1, len(acf_array))]
    fig.append_trace(go.Scatter(x=x_, y=acf_array[1:], mode='markers', marker_color='#1f77b4',
                   marker_size=8), row=row, col=column)
    #PACF
    row, column = (2,1) if vertical else (1,2)
    [fig.append_trace(go.Scatter(x=(x,x), y=(0,pacf_array[x]), mode='lines',line_color='#3f3f3f'), row=row, col=column) 
     for x in range(1, len(pacf_array))]
    fig.append_trace(go.Scatter(x=x_, y=pacf_array[1:], mode='markers', marker_color='#1f77b4',
                   marker_size=8), row=row, col=column)
    fig.update_traces(showlegend=False)
    fig.update_yaxes(zerolinecolor='#000000')
    fig.update_layout(
            autosize=False,
            width=figsize[1],
            height=figsize[0],
            title={
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            title_font={
                "size": 20
            },
            legend_title = None,
            yaxis=dict(
                title_font=dict(size=12),
            ),
            xaxis=dict(
                title_font=dict(size=12),
            )
        )
    return fig
