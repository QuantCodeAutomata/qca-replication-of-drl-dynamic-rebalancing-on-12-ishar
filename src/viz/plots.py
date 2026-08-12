import os
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True


def plot_portfolio_value(
    portfolio_values: Dict[str, pd.Series],
    title: str = 'Portfolio Value (USD)',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cumulative portfolio value over time for multiple strategies.

    Each strategy is rendered as a distinct coloured line using the ``tab10``
    colour cycle so that up to ten strategies can be distinguished without
    manual colour assignment.

    Parameters
    ----------
    portfolio_values : Dict[str, pd.Series]
        Mapping from strategy name to a time-indexed Series of portfolio values
        in USD.  All series are plotted on the same axes.
    title : str
        Chart title.  Default ``'Portfolio Value (USD)'``.
    save_path : str, optional
        If provided, the figure is saved to this file path (format inferred
        from the extension, e.g. ``.png`` or ``.pdf``).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object, ready for further customisation or
        display in a notebook.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(portfolio_values)))
    for (name, series), color in zip(portfolio_values.items(), colors):
        ax.plot(series.index, series.values, label=name, color=color, linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (USD)')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_dynamic_weights(
    weights_df: pd.DataFrame,
    tickers: List[str],
    title: str = 'Dynamic Portfolio Weights',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Stacked area plot of portfolio weights over time.

    The stacked areas sum to 1 at every point in time (assuming the portfolio
    is fully invested).  Up to 12 assets can be labelled with reasonable
    legend clarity; for larger universes consider grouping assets.

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame of shape ``(T, n_assets)`` with asset weights in each column,
        indexed by date.  Columns must correspond to ``tickers`` in order.
    tickers : List[str]
        Asset ticker labels matching the columns of ``weights_df``.
    title : str
        Chart title.  Default ``'Dynamic Portfolio Weights'``.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(weights_df.index, weights_df.values.T, labels=tickers, alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_lambda_horizon(
    lambda_series: pd.Series,
    horizon_series: pd.Series,
    title: str = 'Dynamic Lambda and Horizon Choices',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel scatter plot of the DRL agent's lambda and horizon decisions.

    The upper panel shows the risk-aversion index λₜ selected by the agent at
    each trading day; the lower panel shows the rebalancing horizon hₜ (number
    of days until the next rebalance).

    Parameters
    ----------
    lambda_series : pd.Series
        Time-indexed Series of selected lambda (frontier index) values.
    horizon_series : pd.Series
        Time-indexed Series of selected horizon hₜ values (in trading days).
    title : str
        Overall figure title shown above the upper panel.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.scatter(lambda_series.index, lambda_series.values, s=10, alpha=0.7, color='blue')
    ax1.set_ylabel('Lambda (frontier index)')
    ax1.set_title(title)
    ax2.scatter(horizon_series.index, horizon_series.values, s=10, alpha=0.7, color='red')
    ax2.set_ylabel('Horizon h_t (days)')
    ax2.set_xlabel('Date')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_performance_table(
    metrics_df: pd.DataFrame,
    title: str = 'Performance Metrics',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a metrics DataFrame as a formatted matplotlib table.

    Floating-point values are displayed to four decimal places; all other
    types are converted to strings.  The figure height scales linearly with
    the number of rows so the table remains readable for any number of metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame whose rows are metric names and columns are strategy names
        (or vice versa).  Both orientations are acceptable; the axes labels
        are used as row/column headers.
    title : str
        Chart title rendered above the table.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(16, max(4, len(metrics_df) * 0.5)))
    ax.axis('off')
    col_labels = list(metrics_df.columns)
    row_labels = list(metrics_df.index)
    cell_text = [
        [f'{v:.4f}' if isinstance(v, float) else str(v) for v in row]
        for row in metrics_df.values
    ]
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax.set_title(title, fontsize=12, pad=20)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_sensitivity_tornado(
    sensitivity_df: pd.DataFrame,
    metric: str = 'sharpe_ratio',
    title: str = 'Sensitivity Analysis',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Tornado (horizontal bar) chart for sensitivity analysis.

    Each bar represents the change in ``metric`` induced by varying one
    parameter from its baseline; positive changes are coloured green and
    negative changes are coloured red, following the standard tornado
    convention.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        DataFrame indexed by parameter names with a column named ``metric``
        containing the signed change in that metric for each parameter.
    metric : str
        Name of the metric column to plot.  Default ``'sharpe_ratio'``.
    title : str
        Chart title.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sensitivity_df))
    colors = ['green' if x > 0 else 'red' for x in sensitivity_df[metric]]
    ax.barh(y_pos, sensitivity_df[metric], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sensitivity_df.index)
    ax.set_xlabel(f'Change in {metric}')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, bbox_inches='tight')
    return fig
