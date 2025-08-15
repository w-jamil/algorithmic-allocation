
# Portfolio Strategy Engine & Algorithm Comparison Framework

This project provides a robust framework for backtesting and systematically comparing multiple portfolio weighting algorithms across different asset classes. It is designed for quantitative researchers and developers looking to implement and evaluate strategies based on online machine learning and game-theoretic concepts.

The core of the project is a modular engine that separates data loading, algorithm implementation, and portfolio construction, making it easy to extend and test new ideas. The data and portfolio engine are inspired by best practices from the quantitative finance industry.

## Key Features

-   **Algorithm Implementations:** Includes several learning algorithms:
    -   Aggregating Algorithm (AA)
    -   Super Expert Aggregating Algorithm (SEAA)
    -   Follow The Leader (FTL)
    -   Hedge (Exponentially Weighted Average)
    -   A placeholder for a complex "Foundation Model"
-   **Systematic Comparison:** The main script automatically runs all registered algorithms across multiple asset classes (FX, Rates, Equity) for a direct performance comparison.
-   **Realistic Portfolio Construction:** The `portfolio_engine` simulates a realistic investment process, including:
    -   Daily volatility targeting.
    -   Optional risk parity weighting.
    -   Transaction cost.
-   **Extensible by Design:** Easily add your own custom weighting algorithms by extending the base class and registering them in the main script.
-   **Clear Reporting:** Automatically generates and prints summary tables (Sharpe Ratio, Mean Return, Volatility) and visualises performance with cumulative wealth plots for each asset class.

## Project Structure

The project is organized into clear, single-responsibility modules:

-   **`main.py`**: The main analysis script and entry point. **This is the only file you need to run.** It orchestrates the data loading, algorithm execution, and reporting.
-   **`algorithms.py`**: A dedicated module containing the implementation of all weighting algorithms. New algorithms should be added here.
-   **`portfolio_engine.py`**: The engine for portfolio construction, risk management, and cost calculation.
-   **`data_loader.py`**: A centralized module for loading all required data from pickle files.

### Dependencies
You will need Python 3.9+ and the following libraries. You can install them using pip:

```bash
pip install pandas numpy matplotlib
```
### Data Files

This framework expects four pre-processed data files in pickle format (.pkl) to be present in the same directory as the Python scripts.

**`returns.pkl`**: A dictionary of DataFrames containing asset returns for each class.

**`signal_risk_adj.pkl`**: A dictionary of DataFrames containing the predictive signals from various "experts" for each asset.

**`t_cost.pkl`**: A dictionary of Series containing the estimated transaction costs for each asset.

**`holding_period.pkl`**: A dictionary of Series specifying the smoothing window (in days) for the final conviction signal for each asset.

### How to Run the Full Analysis

No command-line arguments are needed. Simply execute the main script from your terminal:

```bash
python main.py
```

The script will print its progress, display the final performance tables in the console, and then show the cumulative wealth plots.

### Adding a New Algorithm

The framework is designed to be easily extended. To add your own custom algorithm, follow these two steps:

Step 1: Implement the Algorithm in `algorithms.py`

Create a new class in algorithms.py that inherits from the `BaseWeightingAlgorithm` class. You must implement the _run_update_loop method. In `algorithms.py`:

```bash
class MyNewAlgorithm(BaseWeightingAlgorithm):
    """A brief description of your new algorithm."""
    def __init__(self, my_param=0.5, **kwargs):
        super().__init__(name="MyNewAlgo", **kwargs)
        self.my_param = my_param

    def _run_update_loop(self, experts, actual):
        # Your custom weight-updating logic goes here
        # ...
        # This method must return a numpy array of weights
        # with shape (num_timesteps, num_experts)
        pass
```

Step 2: Register the New Algorithm in `main.py`

Open `main.py` and make two small changes. Import your new class:

```bash
from algorithms import (
    AA_Algorithm, 
    SEAA_Algorithm, 
    FollowTheLeader, 
    HedgeAlgorithm, 
    FoundationModel_Placeholder,
    MyNewAlgorithm  # <-- ADD THIS LINE
)
```
Add it to the `**ALGORITHM_FACTORY dictionary**`:
This makes the main script aware of your new algorithm and tells it how to initialize it.

```bash
ALGORITHM_FACTORY = {
    "AA": lambda ac: AA_Algorithm(),
    "SEAA": lambda ac: SEAA_Algorithm(**TUNED_PARAMS.get(ac, {})),
    "FTL": lambda ac: FollowTheLeader(),
    "Hedge": lambda ac: HedgeAlgorithm(eta=2.0),
    "Foundation": lambda ac: FoundationModel_Placeholder(),
    "MyNewAlgo": lambda ac: MyNewAlgorithm(my_param=0.5) } #<-- ADD THIS LINE
```