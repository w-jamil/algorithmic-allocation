import pandas as pd

def get_returns():
    """Loads natural returns from the 'returns.pkl' pickle file."""
    return pd.read_pickle('returns.pkl')

def get_risk_adj_signals():
    """Loads risk-adjusted signals from the 'signal_risk_adj.pkl' pickle file."""
    return pd.read_pickle('signal_risk_adj.pkl')

def get_signal_raw():
    """Loads raw signals from the 'signal_raw.pkl' pickle file."""
    return pd.read_pickle('signal_raw.pkl')

def get_t_cost():
    """Loads transaction cost data from the 't_cost.pkl' pickle file."""
    return pd.read_pickle('t_cost.pkl')

def get_holding_period():
    """Loads holding period data from the 'holding_period.pkl' pickle file."""
    return pd.read_pickle('holding_period.pkl')