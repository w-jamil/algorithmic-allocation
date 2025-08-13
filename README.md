# Portfolio Strategy Engine & Algorithm Comparison Framework

This project provides a framework for testing and systematically comparing multiple portfolio weighting algorithms across different asset classes. Data and portfolio engine is due to the quants I interacted with during the time in industry.

## Project Structure

- **`main.py`**: The main analysis script. **This is the only file you need to run.** It automatically iterates through all defined algorithms and asset classes.
- **`algorithms.py`**: A dedicated module containing the implementation of all weighting algorithms (e.g., AA, SEAA, FTL, Hedge). New algorithms should be added here.
- **`portfolio_engine.py`**: The core mathematical engine for portfolio construction, risk management, and cost calculation.
- **`data_loader.py`**: A centralized module for loading all required data from pickle files.

## Prerequisites

Ensure your data files are present in the same directory as the Python scripts. The required files are:
- `returns.pkl`
- `signal_risk_adj.pkl`
- `t_cost.pkl`
- `holding_period.pkl`

## How to Run the Full Analysis

Simply execute the main script from your terminal. No command-line arguments are needed.

```bash
python main.py