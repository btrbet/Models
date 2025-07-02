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
