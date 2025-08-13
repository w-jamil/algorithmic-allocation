import pandas as pd
import numpy as np
from scipy import stats as sps

# --- UTILITY FUNCTIONS ---

def my_shift(pd_obj, periods=1, extend=False, freq=None):
    """Custom shift function for pandas objects."""
    if extend:
        pd_obj = pd_obj.copy()
        for new_dt in pd.date_range(start=pd_obj.index[-1], freq=freq, periods=periods + 1)[1:]:
            pd_obj.loc[new_dt] = np.nan
    return pd_obj.shift(periods)

def my_ffill(pd_obj, stop=True, limit=None):
    """Custom forward-fill function that only fills NaNs between the first and last valid data points."""
    if stop:
        pd_obj = pd_obj.copy()
        from_idx = pd_obj.first_valid_index()
        to_idx = pd_obj.last_valid_index()
        if from_idx is not None and to_idx is not None:
             pd_obj.loc[from_idx: to_idx] = pd_obj.loc[from_idx: to_idx].ffill(limit=limit)
        return pd_obj
    else:
        return pd_obj.ffill(limit=limit)

# --- CORE PORTFOLIO LOGIC FUNCTIONS ---

def _rp_weights(conviction_df, natural_ret_df, window, min_window, return_lag=1):
    """Calculates Risk Parity (volatility-normalized) weights."""
    natural_ret_df = my_shift(natural_ret_df, periods=return_lag, extend=True, freq='B')
    vol_norm_scaling_df = 1 / natural_ret_df.ewm(span=window, min_periods=min_window).std()
    port_weight_df = vol_norm_scaling_df.mul(conviction_df, axis=1)
    port_weight_df = port_weight_df.dropna(how='all', axis=0)
    return port_weight_df

def _vol_scale_weights(weights_df, natural_ret_df, port_sigma_tgt_s, window, min_window, return_lag=1, long_corr=False):
    """Calculates a scaling factor to adjust portfolio weights to match a target volatility."""
    num_dates = weights_df.shape[0]
    if num_dates == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
        
    num_assets = weights_df.shape[1]
    weights_df = weights_df.sort_index(axis=1)
    natural_ret_df = natural_ret_df[weights_df.columns].sort_index(axis=1)
    natural_ret_df = my_shift(natural_ret_df, periods=return_lag, extend=True, freq='B')

    cov_mdf = natural_ret_df.ewm(span=window, min_periods=min_window).cov()
    
    # Align indices and drop dates that don't exist in the covariance matrix
    common_index = weights_df.index.intersection(cov_mdf.index.get_level_values(0))
    if common_index.empty:
        return pd.Series(dtype=float, index=weights_df.index), pd.Series(dtype=float, index=weights_df.index)

    weights_aligned = weights_df.loc[common_index]
    cov_aligned = cov_mdf.loc[common_index]
    
    num_dates_aligned = weights_aligned.shape[0]
    cov = cov_aligned.values.reshape(num_dates_aligned, num_assets, num_assets)
    weights = weights_aligned.values

    a_arr = np.matmul(weights.reshape(num_dates_aligned, 1, num_assets), cov)
    port_var = np.matmul(a_arr, weights.reshape(num_dates_aligned, num_assets, 1)).reshape(num_dates_aligned)
    port_std = np.sqrt(port_var)
    port_std_s = pd.Series(port_std, index=weights_aligned.index)
    port_std_s = port_std_s.replace({np.inf: np.nan, -np.inf: np.nan}).ffill()
    
    vol_scalar_s = port_sigma_tgt_s.loc[weights_aligned.index] / port_std_s
    return vol_scalar_s.reindex(weights_df.index).ffill(), port_std_s.reindex(weights_df.index).ffill()

def calculate(weights_df, natural_ret_df, var_tgt_daily_pct=None, risk_parity=True, t_cost_s=None, **kwargs):
    """The main engine for constructing a portfolio from a set of weights."""
    out_dd = {'original_weight': weights_df.copy()}
    
    if risk_parity:
        weights_df = _rp_weights(weights_df, natural_ret_df, **kwargs)
        out_dd['rp_weight'] = weights_df.copy()

    if var_tgt_daily_pct is not None:
        if isinstance(var_tgt_daily_pct, (int, float)):
            port_sigma_tgt_s = pd.Series(var_tgt_daily_pct / sps.norm.ppf(0.975), index=weights_df.index)
        else:
            port_sigma_tgt_s = var_tgt_daily_pct / sps.norm.ppf(0.975)
        
        vol_scalar_s, port_std_s = _vol_scale_weights(weights_df, natural_ret_df, port_sigma_tgt_s, **kwargs)
        out_dd['vol_scalar'] = vol_scalar_s
        out_dd['port_std_s'] = port_std_s
        
        vol_scalar_s = vol_scalar_s.replace([np.inf, -np.inf], 0).fillna(0)
        weights_df = weights_df.mul(vol_scalar_s, axis=0)
        out_dd['vs_weight'] = weights_df.copy()

    weights_df = my_ffill(weights_df, stop=True).fillna(0)
    out_dd['tgt_weight'] = weights_df.copy()

    common_idx = weights_df.index.intersection(natural_ret_df.index)
    weights_shifted = weights_df.shift(kwargs.get('return_lag', 1)).loc[common_idx]
    returns_aligned = natural_ret_df.loc[common_idx]
    
    port_ret_df = weights_shifted.mul(returns_aligned)
    out_dd['port_ret'] = port_ret_df

    if t_cost_s is not None:
        t_costs_df = weights_df.diff().abs().mul(t_cost_s)
        out_dd['t_cost'] = t_costs_df
        out_dd['t_port_ret'] = port_ret_df - t_costs_df.reindex(port_ret_df.index).fillna(0)
    else:
        out_dd['t_port_ret'] = port_ret_df

    return out_dd