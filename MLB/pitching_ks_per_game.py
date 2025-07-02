#!/usr/bin/env python3

from pybaseball import  playerid_lookup
from pybaseball import get_splits
import argparse
from scipy.stats import poisson, nbinom
import math


def find_player(first, last):
    """
    Finds a player by their first and last name.

    Parameters:
    - first (str): The player's first name.
    - last (str): The player's last name.

    Returns:
    - player (dict): The player's information.
    """
    player = playerid_lookup(last, first)
    return player

def get_pitching_splits_for_player_for_year(player_bbref_id, year):
    """
    Gets the pitching splits for a player for a given year.

    Parameters:
    - player_bbref_id (str): The player's Baseball-Reference ID.
    - year (int): The year to get the splits for.

    Returns:
    - splits (DataFrame): The pitching splits for the player.
    """
    splits = get_splits(player_bbref_id, year=year, pitching_splits=True)
    return splits

def convert_prob_to_american_odds(prob):
    """
    Converts a probability to American odds.

    Parameters:
    - prob (float): The probability to convert.

    Returns:
    - odds (int): The American odds.
    """
    if prob > 0.5:
        odds = -100.00 / (1 - prob) + 100.00
    else:
        odds = (100.00 / prob) - 100.00
    return int(round(odds))

def predict_strikeout_probability(data, line, use_weights=True):
    """
    Predicts the probability that the given pitcher throws fewer than `line` strikeouts
    in his next game using a weighted Bayesian model with Poisson-Gamma conjugacy.

    Parameters:
    - data (dict): The data to use for the prediction.
    - line (float): The strikeout count to compare against (e.g. 4.5).
    - use_weights (bool): If True, apply recency weights to the data.

    Returns:
    - prob_under (float): P(K < line) for next game.
    - prob_over (float): P(K > line) for next game.
    """

    # Define weights
    weights = {
        '7d': 1.00,
        '14d': 0.75,
        '28d': 0.50,
        'season': 0.25
    }

    # Compute weighted Ks and games
    weighted_Ks = 0.0
    weighted_Games = 0.0

    for period, stats in data.items():
        w = weights[period] if use_weights else 1.0
        weighted_Ks += w * stats['K']
        weighted_Games += w * stats['G']

    # Prior: weakly informative or flat prior
    alpha_0 = 1
    beta_0 = 1

    # Posterior parameters
    alpha_post = alpha_0 + weighted_Ks
    beta_post = beta_0 + weighted_Games

    # Predictive: Negative Binomial(r=alpha, p=beta / (beta + 1))
    r = alpha_post
    p = beta_post / (beta_post + 1)

    # Since strikeouts are discrete, P(K < threshold) = CDF(floor(threshold - 1))
    max_k = int(line - 1)
    prob_under = nbinom.cdf(max_k, r, p)
    prob_over = 1 - prob_under

    return prob_under, prob_over


def main():
    YEAR = 2025
    parser = argparse.ArgumentParser(description="Get pitching stats for a player")
    parser.add_argument("first", help="First name of the player")
    parser.add_argument("last", help="Last name of the player")
    parser.add_argument("--line", help="The current K/G line hung for the player", type=float)
    args = parser.parse_args()

    player = find_player(args.first, args.last)
    splits, info_dict = get_pitching_splits_for_player_for_year(player['key_bbref'][0], YEAR)
    filtered = splits.loc[("Season Totals")]
    print(filtered)
    print("")
    current_season_so = filtered.loc[(str(YEAR) + ' Totals')]['SO']
    current_season_g = filtered.loc[(str(YEAR) + ' Totals')]['G']
    season_so_per_game = float(current_season_so) / float(current_season_g)
    print(f"{args.first} {args.last} has {season_so_per_game} SO/G this season")

    last_28_days_so = filtered.loc[("Last 28 days")]['SO']
    last_28_days_g = filtered.loc[("Last 28 days")]['G']
    last_28_days_so_per_game = float(last_28_days_so) / float(last_28_days_g)
    print(f"{args.first} {args.last} had {last_28_days_so_per_game} SO/G in the last 28 days")

    last_14_days_so = filtered.loc[("Last 14 days")]['SO']
    last_14_days_g = filtered.loc[("Last 14 days")]['G']
    last_14_days_so_per_game = float(last_14_days_so) / float(last_14_days_g)
    print(f"{args.first} {args.last} had {last_14_days_so_per_game} SO/G in the last 14 days")

    last_7_days_so = filtered.loc[("Last 7 days")]['SO']
    last_7_days_g = filtered.loc[("Last 7 days")]['G']
    last_7_days_so_per_game = float(last_7_days_so) / float(last_7_days_g)
    print(f"{args.first} {args.last} had {last_7_days_so_per_game} SO/G in the last 7 days")

    if args.line is not None:
        # Round the line up to the nearest integer
        rounded_line = math.ceil(float(args.line))
        print("")
        print("Using Poisson distribution:")

        # Calculate the probability of getting > {line} SO in the next game
        l28_prob_gt_line = round(1 - poisson.cdf(rounded_line, last_28_days_so_per_game), 4)
        season_prob_gt_line = round(1 - poisson.cdf(rounded_line, season_so_per_game), 4)
        l28_gt_odds = convert_prob_to_american_odds(l28_prob_gt_line)
        season_gt_odds = convert_prob_to_american_odds(season_prob_gt_line)

        print(f"Over {args.line} SO in the next game:")
        print(f"Probability of o{args.line} SO in the next game based on last 28 days: {(l28_prob_gt_line * 100.00):.2f}% ≈ {l28_gt_odds}")
        print(f"Probability of o{args.line} SO in the next game based on season: {(season_prob_gt_line * 100.00):.2f}% ≈ {season_gt_odds}")
        print("")

        # Calculate the probability of getting < {line} SO in the next game
        l28_prob_leq_line = round(poisson.cdf(rounded_line, last_28_days_so_per_game), 4)
        season_prob_leq_line = round(poisson.cdf(rounded_line, season_so_per_game), 4)
        l28_leq_odds = convert_prob_to_american_odds(l28_prob_leq_line)
        season_leq_odds = convert_prob_to_american_odds(season_prob_leq_line)
    

        print(f"Under {args.line} SO in the next game:")
        print(f"Probability of u{args.line} SO in the next game based on last 28 days: {(l28_prob_leq_line * 100.00):.2f}% ≈ {l28_leq_odds}")
        print(f"Probability of u{args.line} SO in the next game based on season: {(season_prob_leq_line * 100.00):.2f}% ≈ {season_leq_odds}")
        print("")

        # Bayesian model with recency weights
        print("Bayesian model:")
        data = {
        '7d':  {'G': last_7_days_g,  'K': last_7_days_so},
        '14d': {'G': last_14_days_g,  'K': last_14_days_so},  # last 14d (not including 7d)
        '28d': {'G': last_28_days_g,  'K': last_28_days_so},  # last 28d (not including 14d)
        'season': {'G': current_season_g, 'K': current_season_so}  # rest of season beyond 28d
    }
        prob_under_line, prob_over_line = predict_strikeout_probability(data, rounded_line)
        prob_under_odds = convert_prob_to_american_odds(prob_under_line)
        prob_over_odds = convert_prob_to_american_odds(prob_over_line)
        print(f"Probability of o{args.line} SO in the next game: {(prob_over_line * 100.00):.2f}% ≈ {prob_over_odds}")
        print(f"Probability of u{args.line} SO in the next game: {(prob_under_line * 100.00):.2f}% ≈ {prob_under_odds}")
        print("")

if __name__ == "__main__":
    main()
