# MLB Python Models

This directory contains Python scripts and models for analyzing and predicting outcomes in Major League Baseball (MLB) betting markets. These tools are designed for bettors, analysts, and content creators who want to explore MLB data and generate actionable insights.

## Included Scripts

### 1. `pitching_ks_per_game.py`
Predicts the probability of a pitcher going over or under a given strikeout (K) line in their next game. The script uses:
- Recent and season-long strikeout data
- Poisson and Bayesian models (with recency weighting)
- American odds conversion for probability outputs

**Features:**
- Looks up a pitcher by name
- Fetches recent and season strikeout stats
- Calculates SO/G (strikeouts per game) for various timeframes
- Predicts probability of going over/under a user-specified K line
- Outputs probabilities and equivalent American odds

**Example usage:**
```bash
./pitching_ks_per_game.py FIRST LAST --line 5.5
```
Replace `FIRST` and `LAST` with the player's name. The `--line` argument is the strikeout line you want to evaluate.

### 2. `./pitching_ks_per_game_by_game.py`
Predicts the probability of a pitcher going or over or under a given strikeout (K) line in their next game. The script uses
a Bayesian Poisson Generalized Linear Model (GLM). By default, the per-game data used as features in the GLM are "baseonballs", "hits", and "inningspitched" for each game thus far in the 2025 season by the given pitcher.

**Example usage:**
```bash
./pitching_ks_per_game_by_game.py FIRST LAST --line 5.5
```
Replace `FIRST` and `LAST` with the player's name. The `--line` argument is the strikeout line you want to evaluate.


#### Backtesting the model

You can use the `--backtest` argument in `pitching_ks_per_game_by_game.py` to evaluate the model's performance across multiple pitchers automatically. This mode simulates betting on every game at both the best closing line and best opening line across four sportsbooks (DraftKings, FanDuel, BetMGM, Caesars) for a list of players. By modifying the feature columns used in the backtest and outputs summary statistics such as win rate, ROI, Sharpe ratio, and more.

**How to use:**

1. **Prepare a CSV file** listing the players you want to backtest. The file should have columns named `player_first_name` and `player_last_name`. Example:

    ```
    player_first_name,player_last_name
    clay,holmes
    kyle,hendricks
    ...
    ```

2. **Run the script with the `--backtest` argument:**

    ```bash
    ./pitching_ks_per_game_by_game.py --backtest path/to/your_players.csv
    ```

    - Replace `path/to/your_players.csv` with the path to your CSV file.

3. **What happens:**  
   - The script will iterate through each player in the CSV, fetch their 2025 game logs, and simulate betting on their strikeout lines using the Bayesian GLM.
   - It will print out performance metrics for both "closing" and "opening" lines, including win/loss/push counts, ROI, Sharpe ratio, and Sortino ratio.

**Note:**  
- You must have a valid OpticOdds API key in your `.env` file for odds fetching and backtesting to work.
- The backtest uses a fixed bet size (1 unit per bet) and grades each bet based on actual game outcomes.

For more details on the CSV format, see the included `backtest_players.csv` example in this directory.

## Dependencies

All scripts require Python 3. Install dependencies with:
```bash
pip install -r requirements.txt
```

**Key packages:**
- [pybaseball](https://github.com/jldbc/pybaseball): For MLB data access
- [scipy](https://scipy.org/): For statistical modeling

## Getting Started

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the desired script as shown above.

## Notes
- These scripts are for educational and illustrative purposes. They may require adaptation for production or betting use.
- For more information about the overall project, see the [top-level README](../README.md).

## License
See the repository's [LICENSE.md](/LICENSE.md) file for details.
