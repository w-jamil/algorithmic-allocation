import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Import from your organized project files
import portfolio_engine
import data_loader
from algorithms import (
    AA_Algorithm, 
    SEAA_Algorithm, 
    FollowTheLeader, 
    HedgeAlgorithm, 
    LLR_Inspired_Hedge, 
    FoundationModel_Placeholder
)

def smooth_holding_period(signal_df, hp_days_s):
    """Applies a rolling mean based on holding period to smooth signals."""
    if isinstance(signal_df, pd.Series):
        signal_df = pd.DataFrame(signal_df)
    smooth_signal_df = pd.DataFrame()
    for name in signal_df.columns:
        if name in hp_days_s.index:
            window_size = int(hp_days_s.get(name, 1))
            smooth_signal_s = signal_df[name].rolling(window_size, min_periods=1).mean()
            smooth_signal_df[name] = smooth_signal_s
    return smooth_signal_df.reindex(signal_df.index)

def run_strategy(asset_class, algorithm):
    """
    Runs a full investment strategy for a given asset class and weighting algorithm.
    """
    print(f"  > Running for Asset Class: '{asset_class.upper()}' using Algorithm: '{algorithm.name}'...")

    # Load Data
    nat_vol_rtn_mdf = all_returns[asset_class]
    signal_mdf = all_signals[asset_class]
    tcost_s = all_tcosts[asset_class]
    hold_s = all_holds[asset_class]

    # Logic to handle the special case of the Foundation Model
    if isinstance(algorithm, FoundationModel_Placeholder):
        conviction_df = algorithm.get_conviction_signal(nat_vol_rtn_mdf, signal_mdf)
    else:
        # Standard two-step process for online learning algorithms
        expert_weights_df = algorithm.get_weights(nat_vol_rtn_mdf, signal_mdf)
        common_idx = expert_weights_df.index.intersection(signal_mdf.index)
        expert_weights_aligned = expert_weights_df.reindex(common_idx)
        signal_mdf_aligned = signal_mdf.reindex(common_idx)
        
        w_signal_components = []
        assets_in_run = expert_weights_aligned.columns.get_level_values(0).unique()
        for asset in assets_in_run:
            if asset in signal_mdf_aligned.columns.get_level_values(0):
                asset_signals = signal_mdf_aligned[asset]
                asset_expert_weights = expert_weights_aligned[asset]
                weighted_asset_signals = asset_signals.mul(asset_expert_weights)
                w_signal_components.append(weighted_asset_signals.sum(axis=1).rename(asset))
        conviction_df = pd.concat(w_signal_components, axis=1)

    # Smooth the final conviction signal
    s_conviction_df = smooth_holding_period(conviction_df, hold_s)

    # Build Portfolio
    portfolio_param_dd = {
        'var_tgt_daily_pct': 0.01, 'risk_parity': True,
        'window': 250, 'min_window': 125, 'return_lag': 1
    }

    port_dd = portfolio_engine.calculate(
        weights_df=s_conviction_df,
        natural_ret_df=nat_vol_rtn_mdf,
        t_cost_s=tcost_s,
        **portfolio_param_dd
    )
    
    return port_dd

if __name__ == '__main__':
    try:
        # --- Configuration: Define all experiments to run ---
        ASSET_CLASSES = ['fx', 'rates', 'equity']
        
        # --- Hyperparameter Tuning Dictionary ---
        # This single dictionary now controls the parameters for BOTH SEAA and FixedShare.
        TUNED_PARAMS = {
            'fx':     {'alpha': 0.2, 'eta': 2.0},
            'rates':  {'alpha': 0.5, 'eta': 1.0},
            'equity': {'alpha': 0.6, 'eta': 0.8}
        }

        # --- Algorithm Factory ---
        # This factory now uses the same TUNED_PARAMS for FixedShare and SEAA.
        ALGORITHM_FACTORY = {
            "AA": lambda ac: AA_Algorithm(),
            "SEAA": lambda ac: SEAA_Algorithm(**TUNED_PARAMS.get(ac, {})),
            "FTL": lambda ac: FollowTheLeader(),
            "Hedge": lambda ac: HedgeAlgorithm(eta=2.0),
            "LLR": lambda ac: LLR_Inspired_Hedge(eta=2.0),
            "Foundation": lambda ac: FoundationModel_Placeholder()
        }

        # --- Data Loading ---
        print("Loading all data...")
        all_returns = data_loader.get_returns()
        all_signals = data_loader.get_risk_adj_signals()
        all_tcosts = data_loader.get_t_cost()
        all_holds = data_loader.get_holding_period()
        print("Data loaded successfully.\n")

        # --- Batch Execution ---
        all_results = []
        print("Starting batch execution of all algorithms across all asset classes...")
        for ac in ASSET_CLASSES:
            for algo_name, algo_constructor in ALGORITHM_FACTORY.items():
                algo = algo_constructor(ac)
                portfolio_results = run_strategy(asset_class=ac, algorithm=algo)
                
                t_port_ret_df = portfolio_results.get('t_port_ret')
                if t_port_ret_df is not None and not t_port_ret_df.empty:
                    port_ret_s = t_port_ret_df.sum(axis=1)
                    
                    trading_days = 252
                    mean_ret = port_ret_s.mean() * trading_days
                    std_dev = port_ret_s.std() * np.sqrt(trading_days)
                    sharpe = mean_ret / std_dev if std_dev != 0 else 0

                    all_results.append({
                        'Asset Class': ac.upper(),
                        'Algorithm': algo.name,
                        'Mean Return': mean_ret,
                        'Std Dev': std_dev,
                        'Sharpe Ratio': sharpe,
                        'Returns': port_ret_s
                    })
        print("\nBatch execution complete.\n")

        # --- Reporting ---
        if not all_results:
            print("No results were generated. Exiting.")
            sys.exit()

        results_df = pd.DataFrame(all_results)
        
        sharpe_pivot = results_df.pivot_table(index='Algorithm', columns='Asset Class', values='Sharpe Ratio')
        # ... (rest of the reporting is the same) ...

        print("--- PERFORMANCE SUMMARY ---")
        print("\n** Annualized Sharpe Ratio **")
        print(sharpe_pivot.to_string(float_format="%.2f"))
        # ... (print other tables) ...

        print("\nGenerating comparison plots...")
        plt.style.use('seaborn-v0_8-darkgrid')
        
        asset_classes_in_results = results_df['Asset Class'].unique()
        for ac_upper in asset_classes_in_results:
            plt.figure(figsize=(15, 8))
            ac_results = [res for res in all_results if res['Asset Class'] == ac_upper]
            
            if not ac_results:
                continue

            for result in ac_results:
                wealth_index = (1 + result['Returns']).cumprod()
                label = f"{result['Algorithm']} (SR: {result['Sharpe Ratio']:.2f})"
                plt.plot(wealth_index.index, wealth_index, label=label)

            plt.legend(title='Algorithm (Sharpe Ratio)', loc='upper left')
            plt.title(f"Cumulative Wealth Comparison for {ac_upper} Asset Class", fontsize=16)
            plt.ylabel("Cumulative Wealth (Log Scale)")
            plt.xlabel("Date")
            plt.yscale('log')
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            plt.show()

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required data file was not found: {e.filename}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()