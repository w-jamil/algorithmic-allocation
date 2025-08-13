# algorithms.py

import pandas as pd
import numpy as np

# --- UTILITY FUNCTIONS ---
# (These remain the same)
def _subs(weights, pred, eta, minY, maxY):
    gmin = -1 / eta * np.log(np.dot(weights, np.exp(pred - (pred - minY) ** 2)))
    gmax = -1 / eta * np.log(np.dot(weights, np.exp(pred - (pred - maxY) ** 2)))
    g = 0.5 * (minY + maxY) - (gmin - gmax) / (2 * (maxY - minY))
    return g

def _robust_normalize(w):
    s = w.sum()
    if s > 1e-9:
        return w / s
    else:
        n = len(w)
        return np.ones(n) / n if n > 0 else w

# --- BASE CLASS FOR ALL ALGORITHMS ---
class BaseWeightingAlgorithm:
    """An abstract base class for all weighting algorithms."""
    def __init__(self, name="Base", eta=2.0, minY=-1.0, maxY=1.0):
        self.name = name
        self.eta = eta
        self.minY = minY
        self.maxY = maxY
        print(f"Initialized Base for: {self.name}")

    def _run_update_loop(self, experts, actual):
        """This is the abstract method that each specific algorithm must implement."""
        raise NotImplementedError("The '_run_update_loop' method must be implemented by subclasses.")

    def get_weights(self, nat_rtn_mdf, signal_mdf):
        """Main public method to orchestrate the weight generation process."""
        # This method's logic is correct and remains the same.
        ac_nat_rtn_mdf = nat_rtn_mdf.copy()
        ac_signal_mdf = signal_mdf.copy().shift(1)
        asset_ls = ac_nat_rtn_mdf.columns.tolist()
        weight_dd = dict()
        for asset in asset_ls:
            if asset not in ac_signal_mdf.columns.get_level_values(0): continue
            asset_rtn_s = ac_nat_rtn_mdf[asset].loc['2010':].dropna()
            asset_sig_df = ac_signal_mdf[asset].loc['2010':].dropna()
            c_idx = sorted(list(set(asset_rtn_s.index).intersection(set(asset_sig_df.index))))
            if not c_idx: continue
            asset_rtn_s = asset_rtn_s.reindex(c_idx)
            asset_sig_df = asset_sig_df.reindex(c_idx)
            w_vals = self._run_update_loop(asset_sig_df.values, asset_rtn_s.values)
            w_df = pd.DataFrame(w_vals, columns=asset_sig_df.columns, index=c_idx)
            weight_dd[asset] = w_df
        if not weight_dd: return pd.DataFrame()
        return pd.concat(weight_dd, axis=1)

    def __str__(self):
        return self.name

# --- SPECIFIC ALGORITHM IMPLEMENTATIONS ---

class AA_Algorithm(BaseWeightingAlgorithm):
    """Implements the standard 'AA' (Aggregating Algorithm)."""
    def __init__(self, **kwargs):
        super().__init__(name="AA", **kwargs)
    def _run_update_loop(self, experts, actual):
        n = np.size(experts, axis=1); t = np.size(experts, axis=0)
        w = np.ones(n); temp = np.zeros((t, n))
        for i in range(t):
            w = _robust_normalize(w)
            temp[i,] = w
            w = w * np.exp(-self.eta * (experts[i,] - actual[i]) ** 2)
        return temp

class SEAA_Algorithm(BaseWeightingAlgorithm):
    """
    Implements your specific 'SEAA' (Specialist Expert Aggregating Algorithm) variant.
    LOGIC FIX: This now uses ONLY the power update rule for performance, making it
    distinct from Hedge/FixedShare which use the exponential rule.
    """
    def __init__(self, alpha=0.2, eta=2.0, **kwargs):
        super().__init__(name="SEAA", eta=eta, **kwargs)
        self.alpha = alpha
        print(f" -> SEAA initialized with alpha={self.alpha}, eta={self.eta}")

    def _run_update_loop(self, experts, actual):
        n = np.size(experts, axis=1); t = np.size(experts, axis=0)
        w = np.ones(n); temp = np.zeros((t, n))
        for i in range(t):
            w = _robust_normalize(w)
            
            # 1. Mix BEFORE the performance update
            if n > 1:
                w = (1 - self.alpha) * w + self.alpha / (n - 1) * (1 - w)
            
            temp[i,] = w
            
            # 2. Apply the POWER update rule. This is now the ONLY performance update.
            loss = (experts[i,] - actual[i]) ** 2
            # Add a small epsilon to prevent 0**loss issues
            w = (w + 1e-9) ** loss
        return temp


class FollowTheLeader(BaseWeightingAlgorithm):
    # (This class remains the same)
    def __init__(self, **kwargs):
        super().__init__(name="FTL", **kwargs)
    def _run_update_loop(self, experts, actual):
        n, t = np.size(experts, axis=1), np.size(experts, axis=0)
        cumulative_loss = np.zeros(n)
        weights_history = np.zeros((t, n))
        for i in range(t):
            best_expert_idx = np.argmin(cumulative_loss)
            w = np.zeros(n)
            w[best_expert_idx] = 1.0
            weights_history[i, :] = w
            loss_t = (experts[i, :] - actual[i]) ** 2
            cumulative_loss += loss_t
        return weights_history

class HedgeAlgorithm(BaseWeightingAlgorithm):
    # (This class remains the same)
    def __init__(self, **kwargs):
        super().__init__(name="Hedge", **kwargs)
    def _run_update_loop(self, experts, actual):
        n, t = np.size(experts, axis=1), np.size(experts, axis=0)
        w = np.ones(n)
        weights_history = np.zeros((t, n))
        for i in range(t):
            w = _robust_normalize(w)
            weights_history[i, :] = w
            loss_t = (experts[i, :] - actual[i]) ** 2
            w = w * np.exp(-self.eta * loss_t)
        return weights_history

class LLR_Inspired_Hedge(BaseWeightingAlgorithm):
    # (This class remains the same)
    def __init__(self, **kwargs):
        super().__init__(name="LLR", **kwargs)
    def _run_update_loop(self, experts, actual):
        n, t = np.size(experts, axis=1), np.size(experts, axis=0)
        w = np.ones(n)
        weights_history = np.zeros((t, n))
        for i in range(t):
            w = _robust_normalize(w)
            weights_history[i, :] = w
            loss_t = (experts[i, :] - actual[i]) ** 2
            w = w * np.exp(-self.eta * loss_t)
        return weights_history
        
class FoundationModel_Placeholder(BaseWeightingAlgorithm):
    # (This class remains the same)
    def __init__(self, momentum_window=20, **kwargs):
        super().__init__(name="Foundation", **kwargs)
        self.momentum_window = momentum_window
    def get_conviction_signal(self, nat_rtn_mdf, signal_mdf):
        momentum_df = signal_mdf.diff(self.momentum_window)
        transposed_momentum = momentum_df.T
        transposed_momentum.index = pd.MultiIndex.from_tuples(transposed_momentum.index)
        best_signal_df = transposed_momentum.groupby(level=0).idxmax().T
        final_conviction = pd.DataFrame(index=signal_mdf.index, columns=nat_rtn_mdf.columns, dtype=float)
        for asset in final_conviction.columns:
            if asset in best_signal_df.columns:
                best_signals_for_asset = best_signal_df[asset].dropna()
                if not best_signals_for_asset.empty:
                    row_indices = signal_mdf.index.get_indexer(best_signals_for_asset.index)
                    col_indices = signal_mdf.columns.get_indexer(best_signals_for_asset.values)
                    lookup_values = signal_mdf.values[row_indices, col_indices]
                    final_conviction.loc[best_signals_for_asset.index, asset] = lookup_values
        final_conviction = final_conviction.infer_objects(copy=False).ffill().fillna(0.0)
        return final_conviction.astype(float)