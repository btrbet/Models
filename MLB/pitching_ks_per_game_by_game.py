#!/usr/bin/env python3

import argparse
from bdb import Breakpoint
from scipy.stats import poisson, nbinom
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az
import logging
from utils import (
    find_player,
    convert_prob_to_american_odds,
    get_pitching_game_log_df,
    calculate_average_strikeouts_per_game,
)
from opticodds import get_fixture_id, fetch_odds
from collections import Counter
import csv
import pprint

# Turn off pymc logging
logging.getLogger("pymc").setLevel(logging.WARNING)

# Turn off pybaseball logging
logging.getLogger("pybaseball").setLevel(logging.WARNING)


def plot_strikeouts_per_game(df):
    """
    Plots the number of strikeouts in each game.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least 'date' and 'strikeouts' columns
    """
    if "date" not in df.columns or "strikeouts" not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'strikeouts' columns.")

    # Convert date to datetime if not already
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Calculate average strikeouts per game
    avg_strikeouts = calculate_average_strikeouts_per_game(df)
    avg_strikeouts_trailing_7 = calculate_average_strikeouts_per_game(
        df, trailing_days=7
    )
    avg_strikeouts_trailing_14 = calculate_average_strikeouts_per_game(
        df, trailing_days=14
    )
    avg_strikeouts_trailing_30 = calculate_average_strikeouts_per_game(
        df, trailing_days=30
    )
    avg_strikeouts_trailing_60 = calculate_average_strikeouts_per_game(
        df, trailing_days=60
    )
    avg_strikeouts_trailing_90 = calculate_average_strikeouts_per_game(
        df, trailing_days=90
    )

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["strikeouts"], marker="o", linestyle="-")
    plt.xlabel("Date")
    plt.ylabel("Strikeouts")
    plt.title("Strikeouts By Game")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Rainbow order: red, orange, yellow, green, blue, indigo
    avg_lines = [
        ("Average (season)", avg_strikeouts, "red"),
        ("Average (90 days)", avg_strikeouts_trailing_90, "orange"),
        ("Average (60 days)", avg_strikeouts_trailing_60, "yellow"),
        ("Average (30 days)", avg_strikeouts_trailing_30, "green"),
        ("Average (14 days)", avg_strikeouts_trailing_14, "blue"),
        ("Average (7 days)", avg_strikeouts_trailing_7, "indigo"),
    ]
    max_date = df["date"].max()
    for i, (label, avg, color) in enumerate(avg_lines):
        if i == 0:
            # Season-long average: full width
            plt.axhline(y=avg, color=color, linestyle="--", label=f"{label}: {avg:.2f}")
        else:
            # Trailing averages: horizontal lines from start_date to max_date
            days = [90, 60, 30, 14, 7][i - 1]
            start_date = max_date - pd.Timedelta(days=days)
            plt.hlines(
                y=avg,
                xmin=start_date,
                xmax=max_date,
                colors=color,
                linestyles="--",
                label=f"{label}: {avg:.2f}",
            )

    # Add herringbone (crosshatch) pattern between overlapping average lines
    hatch_params = [
        (90, 60, "xx"),  # (longer, shorter, hatch)
        (60, 30, "xx"),
        (30, 14, "xx"),
        (14, 7, "xx"),
    ]
    avg_map = {
        90: avg_strikeouts_trailing_90,
        60: avg_strikeouts_trailing_60,
        30: avg_strikeouts_trailing_30,
        14: avg_strikeouts_trailing_14,
        7: avg_strikeouts_trailing_7,
    }

    ax = plt.gca()
    for longer, shorter, hatch in hatch_params:
        y1 = avg_map[longer]
        y2 = avg_map[shorter]
        x_start = max_date - pd.Timedelta(days=shorter)
        x_end = max_date

        # Create x values for the overlap region
        n_points = 200
        x_vals = pd.date_range(start=x_start, end=x_end, periods=n_points)
        ax.fill_between(
            x_vals,
            y1,
            y2,
            facecolor="none",
            edgecolor="gray",
            hatch=hatch,
            linewidth=0.0,
            zorder=2,
            alpha=0.7,
        )

    plt.legend()
    plt.show()


def predict_strikeout_over_probability(
    data: pd.DataFrame, N: float, show_plots: bool = False
):
    """
    Predict the probability that a pitcher will have more than N strikeouts
    in the next game using Bayesian inference with a Poisson GLM.

    Parameters:
        data (pd.DataFrame): Must include a 'strikeouts' column. All other columns are used as features.
        N (float): The sportsbook over/under line (e.g., 6.5)
        show_plots (bool): Whether to show posterior coefficient plots

    Returns:
        prob_over (float): Posterior predictive probability that strikeouts > N
        odds (int): American odds for the over/under line
        coef_samples_dict (dict): Dictionary of posterior samples for each feature coefficient
    """
    # Separate features and target
    y = data["strikeouts"].dropna().astype(float)
    X = data.drop(columns=["strikeouts"]).loc[y.index]
    X = X.astype(float)
    feature_names = X.columns.tolist()
    X_values = X.values
    n_features = X.shape[1]

    # Use the most recent row as the 'next game' features
    X_new = X.iloc[[-1]].values  # shape (1, n_features)

    with pm.Model() as model:
        # Priors for coefficients
        beta = pm.Normal("beta", mu=0, sigma=5, shape=n_features)
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        # Linear predictor
        mu = pm.math.exp(intercept + pm.math.dot(X_values, beta))
        # Likelihood
        obs = pm.Poisson("obs", mu=mu, observed=y)
        # Posterior sampling
        trace = pm.sample(
            1000,
            tune=1000,
            cores=1,
            progressbar=False,
            random_seed=42,
            target_accept=0.95,
        )
        # Extract posterior samples as numpy arrays
        intercept_samples = trace.posterior["intercept"].values.flatten()  # (n_draws,)
        beta_samples = trace.posterior["beta"].values  # (chain, draw, n_features)
        beta_samples = beta_samples.reshape(
            -1, beta_samples.shape[-1]
        )  # (n_draws, n_features)
        # Compute mu_pred for each posterior sample
        mu_pred = np.exp(intercept_samples + np.dot(beta_samples, X_new.T).flatten())
        y_pred = np.random.poisson(mu_pred)

    # Probability that predicted strikeouts > N
    prob_over = np.mean(y_pred > N)
    odds = convert_prob_to_american_odds(prob_over)

    coef_samples_dict = {}
    for i, name in enumerate(feature_names):
        coef_samples = beta_samples[:, i]
        coef_samples_dict[name] = coef_samples
        mean = np.mean(coef_samples)
        median = np.median(coef_samples)
        lower = np.percentile(coef_samples, 2.5)
        upper = np.percentile(coef_samples, 97.5)
        # print(f"Feature: {name}")
        # print(f"  Mean: {mean:.3f}, Median: {median:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
        if show_plots:
            plt.hist(coef_samples, bins=30, alpha=0.5, label=name)
    if show_plots:
        plt.legend()
        plt.title("Posterior distributions of feature coefficients (beta)")
        plt.xlabel("Coefficient value")
        plt.ylabel("Frequency")
        plt.show()

    return prob_over, odds, coef_samples_dict


def predict_strikeout_under_probability(
    data: pd.DataFrame, N: float, show_plots: bool = False
):
    """
    Predict the probability that a pitcher will have less than N strikeouts
    in the next game using Bayesian inference with a Poisson GLM.

    Parameters:
        data (pd.DataFrame): Must include a 'strikeouts' column. All other columns are used as features.
        N (float): The sportsbook over/under line (e.g., 6.5)
        show_plots (bool): Whether to show posterior coefficient plots

    Returns:
        prob_under (float): Posterior predictive probability that strikeouts < N
        odds (int): American odds for the under
        coef_samples_dict (dict): Dictionary of posterior samples for each feature coefficient
    """
    # Separate features and target
    y = data["strikeouts"].dropna().astype(float)
    X = data.drop(columns=["strikeouts"]).loc[y.index]
    X = X.astype(float)
    feature_names = X.columns.tolist()
    X_values = X.values
    n_features = X.shape[1]

    # Use the most recent row as the 'next game' features
    X_new = X.iloc[[-1]].values  # shape (1, n_features)

    with pm.Model() as model:
        # Priors for coefficients
        beta = pm.Normal("beta", mu=0, sigma=5, shape=n_features)
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        # Linear predictor
        mu = pm.math.exp(intercept + pm.math.dot(X_values, beta))
        # Likelihood
        obs = pm.Poisson("obs", mu=mu, observed=y)
        # Posterior sampling
        trace = pm.sample(
            1000,
            tune=1000,
            cores=1,
            progressbar=False,
            random_seed=42,
            target_accept=0.95,
        )
        # Extract posterior samples as numpy arrays
        intercept_samples = trace.posterior["intercept"].values.flatten()  # (n_draws,)
        beta_samples = trace.posterior["beta"].values  # (chain, draw, n_features)
        beta_samples = beta_samples.reshape(
            -1, beta_samples.shape[-1]
        )  # (n_draws, n_features)
        # Compute mu_pred for each posterior sample
        mu_pred = np.exp(intercept_samples + np.dot(beta_samples, X_new.T).flatten())
        y_pred = np.random.poisson(mu_pred)

    # Probability that predicted strikeouts < N
    prob_under = np.mean(y_pred < N)
    odds = convert_prob_to_american_odds(prob_under)

    coef_samples_dict = {}
    for i, name in enumerate(feature_names):
        coef_samples = beta_samples[:, i]
        coef_samples_dict[name] = coef_samples
        mean = np.mean(coef_samples)
        median = np.median(coef_samples)
        lower = np.percentile(coef_samples, 2.5)
        upper = np.percentile(coef_samples, 97.5)
        # print(f"Feature: {name}")
        # print(f"  Mean: {mean:.3f}, Median: {median:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
        if show_plots:
            plt.hist(coef_samples, bins=30, alpha=0.5, label=name)
    if show_plots:
        plt.legend()
        plt.title("Posterior distributions of feature coefficients (beta)")
        plt.xlabel("Coefficient value")
        plt.ylabel("Frequency")
        plt.show()

    return prob_under, odds, coef_samples_dict


def backtest_model_for_player(player: dict, df: pd.DataFrame):
    print(
        f"Backtesting the model for {player['name_first'][0]} {player['name_last'][0]}"
    )

    lines = ["closing", "opening"]
    bets = {"closing": [], "opening": []}

    # 1. Iterate over each game in the data frame df
    for index, row in df.iterrows():
        fixture_id = get_fixture_id(
            row["date"], f"{player['name_first'][0]} {player['name_last'][0]}"
        )
        print(f"\nFixture ID: {fixture_id} on {row['date']}")
        # 2. For each game, fetch the Optic Odds `fixture` for the game using https://developer.opticodds.com/reference/get_fixtures
        # Use the following sportsbooks:
        # - DraftKings
        # - FanDuel
        # - BetMGM
        # - ESPN Bet
        # - Caesars
        odds = fetch_odds(
            fixture_id, f"{player['name_first'][0]} {player['name_last'][0]}"
        )
        for line in lines:
            # 3. Determine the best closing line 'points' for the 'player strikeouts' over/under market
            # Extract all 'points' values from the odds object
            points_list = [
                odds[line][book]["over_points"]
                for book in odds[line].keys()
                if odds[line][book].get("over_points") is not None
            ]

            # Count the frequency of each points value
            points_counter = Counter(points_list)

            # Get the most common points value
            if points_counter:
                best_closing_line_points = points_counter.most_common(1)[0][0]
            else:
                best_closing_line_points = None

            if best_closing_line_points is None or best_closing_line_points == 0:
                continue

            # 4. Determine the best closing line 'price' for the over 'player strikeouts' market
            # The best closing line 'price' is the greatest 'over_price' value across all odds entries
            # that have a 'points' value equal to the best_closing_line_points
            best_closing_line_over_price = max(
                [
                    odds[line][book]["over_price"]
                    for book in odds[line].keys()
                    if odds[line][book]["over_points"] == best_closing_line_points
                ]
            )
            best_over_book = [
                book
                for book in odds[line].keys()
                if odds[line][book]["over_points"] == best_closing_line_points
                and odds[line][book]["over_price"] == best_closing_line_over_price
            ][0]
            # 5. Determine the best closing line 'price' for the under 'player strikeouts' market
            # The best closing line 'price' is the greatest 'under_price' value across all odds entries
            # that have a 'points' value equal to the best_closing_line_points
            best_closing_line_under_price = max(
                [
                    odds[line][book]["under_price"]
                    for book in odds[line].keys()
                    if odds[line][book]["under_points"] == best_closing_line_points
                ]
            )
            best_under_book = [
                book
                for book in odds[line].keys()
                if odds[line][book]["under_points"] == best_closing_line_points
                and odds[line][book]["under_price"] == best_closing_line_under_price
            ][0]

            print(
                f"Best {line} line over price: {best_closing_line_over_price} at {best_over_book}"
            )
            print(
                f"Best {line} line under price: {best_closing_line_under_price} at {best_under_book}"
            )

            # 6. Create a DataFrame with the required columns for prediction using only rows where the 'date' is before the current game's date
            data_for_prediction = df[
                # "date" and "strikeouts" are required columns for the model
                # All other columns are used as features in the GLM
                ["date", "strikeouts", "baseonballs", "hits", "inningspitched"]
            ].copy()
            data_for_prediction = data_for_prediction[
                data_for_prediction["date"] < row["date"]
            ]
            # Remove the 'date' column
            data_for_prediction = data_for_prediction.drop(columns=["date"])

            # Check for an empty data frame
            if data_for_prediction.empty:
                continue

            # 6. Run the model to predict the probability of the given pitcher pitching more than the best closing line 'points'
            prob_over, odds_over, coef_samples_dict_over = (
                predict_strikeout_over_probability(
                    data_for_prediction,
                    best_closing_line_points,
                    show_plots=False,
                )
            )

            # 7. Run the model to predict the probability of the given pitcher pitching less than the best closing line 'points'
            prob_under, odds_under, coef_samples_dict_under = (
                predict_strikeout_under_probability(
                    data_for_prediction,
                    best_closing_line_points,
                    show_plots=False,
                )
            )

            # 8. Compare the over model prediction with the under model prediction and select which one has a higher probability
            # 9. Bet the over or under market based on the selected prediction for 1u.
            price = (
                best_closing_line_over_price
                if prob_over > prob_under
                else best_closing_line_under_price
            )
            amount = 100
            # Calculate win_amount based on American odds
            if price > 0:
                win_amount = amount * (price / 100)
            else:
                win_amount = amount * (100 / abs(price))
            bets[line].append(
                {
                    "date": row["date"],
                    "fixture_id": fixture_id,
                    "player": f"{player['name_first'][0]} {player['name_last'][0]}",
                    "points": best_closing_line_points,
                    "selection": "over" if prob_over > prob_under else "under",
                    "price": price,
                    "book": (
                        best_over_book if prob_over > prob_under else best_under_book
                    ),
                    "prob": (
                        float(f"{prob_over:.2f}")
                        if prob_over > prob_under
                        else float(f"{prob_under:.2f}")
                    ),
                    "amount": amount,
                    "win_amount": win_amount,
                    "result": None,
                }
            )
            # 10. Grade the bet by inspecting the player's actual performance in the game.
            # Find the row in the df where the 'date' is the same as the current game's date
            game_row = df[df["date"] == row["date"]]
            # Get the 'strikeouts' value for the current game
            strikeouts = game_row["strikeouts"].values[0]
            if strikeouts > best_closing_line_points:
                bets[line][-1]["result"] = "win"
            elif strikeouts == best_closing_line_points:
                bets[line][-1]["result"] = "push"
            else:
                bets[line][-1]["result"] = "loss"

    return bets


def main():
    YEAR = 2025
    parser = argparse.ArgumentParser(description="Get pitching stats for a player")
    parser.add_argument("first", nargs="?", help="First name of the player")
    parser.add_argument("last", nargs="?", help="Last name of the player")
    parser.add_argument(
        "--line", help="The current K/G line hung for the player", type=float
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Show posterior coefficient plots"
    )
    parser.add_argument(
        "--backtest",
        type=str,
        help="Path to CSV file with player_first_name and player_last_name columns for backtesting",
        default=None,
    )

    args = parser.parse_args()

    if not args.backtest and (not args.first or not args.last):
        parser.error(
            "the following arguments are required: first, last (unless --backtest is set)"
        )

    if not args.backtest:
        player = find_player(args.first, args.last)
        df = get_pitching_game_log_df(player["key_mlbam"][0], YEAR)

        # Print the average strikeouts per game over the various trailing periods: season, 90 days, 60 days, 30 days, 14 days, 7 days
        print(
            f"Average strikeouts per game (season): {calculate_average_strikeouts_per_game(df)}"
        )
        print(
            f"Average strikeouts per game (90 days): {calculate_average_strikeouts_per_game(df, trailing_days=90)}"
        )
        print(
            f"Average strikeouts per game (60 days): {calculate_average_strikeouts_per_game(df, trailing_days=60)}"
        )
        print(
            f"Average strikeouts per game (30 days): {calculate_average_strikeouts_per_game(df, trailing_days=30)}"
        )
        print(
            f"Average strikeouts per game (14 days): {calculate_average_strikeouts_per_game(df, trailing_days=14)}"
        )
        print(
            f"Average strikeouts per game (7 days): {calculate_average_strikeouts_per_game(df, trailing_days=7)}"
        )

        # Create a DataFrame with the required columns for prediction
        data_for_prediction = df[
            ["strikeouts", "baseonballs", "hits", "inningspitched"]
        ].copy()
        # data_for_prediction = df[["strikeouts", "inningspitched", "whip", "era"]].copy()

        prob_over, odds_over, coef_samples_dict_over = (
            predict_strikeout_over_probability(
                data_for_prediction, args.line, show_plots=args.show_plots
            )
        )
        print(
            f"Probability of over {args.line} K/G in the next game: {prob_over * 100:.2f}% ≈ {odds_over}"
        )

        prob_under, odds_under, coef_samples_dict_under = (
            predict_strikeout_under_probability(
                data_for_prediction, args.line, show_plots=args.show_plots
            )
        )
        print(
            f"Probability of under {args.line} K/G in the next game: {prob_under * 100:.2f}% ≈ {odds_under}"
        )

        if args.show_plots:
            plot_strikeouts_per_game(df)
    else:
        all_bets = {
            "closing": [],
            "opening": [],
        }

        # Read the CSV file of players to backtest
        with open(args.backtest) as csv_file:
            raw_players = [
                {k: v for k, v in row.items()}
                for row in csv.DictReader(csv_file, skipinitialspace=True)
            ]

        for raw_player in raw_players:
            player = find_player(
                raw_player["player_first_name"], raw_player["player_last_name"]
            )

            try:
                mlbam_key = player["key_mlbam"][0]
                df = get_pitching_game_log_df(mlbam_key, YEAR)
            except Exception as e:
                print(
                    f"Error getting game log for {raw_player['player_first_name']} {raw_player['player_last_name']}: {e}"
                )
                continue

            bets = backtest_model_for_player(player, df)
            all_bets["closing"].extend(bets["closing"])
            all_bets["opening"].extend(bets["opening"])

        for line in all_bets.keys():
            bets = all_bets[line]
            # 11. Evaluate the model's performance by calculating the following metrics:
            # Count the number of entries in the bets list where the 'result' is 'win'
            wins = len([bet for bet in bets if bet["result"] == "win"])
            # Count the number of entries in the bets list where the 'result' is 'loss'
            losses = len([bet for bet in bets if bet["result"] == "loss"])
            # Count the number of entries in the bets list where the 'result' is 'push'
            pushes = len([bet for bet in bets if bet["result"] == "push"])
            # Calculate the win rate
            win_rate = (
                (float(wins) / float(wins + losses)) * 100
                if (wins + losses) > 0
                else 0.0
            )

            # Calculate the total amount bet
            total_amount_bet = sum([bet["amount"] for bet in bets])

            # Calculate the total profit (loss)
            total_profit_loss = sum(
                [bet["win_amount"] for bet in bets if bet["result"] == "win"]
            ) - sum([bet["amount"] for bet in bets if bet["result"] == "loss"])

            # Calculate the ROI
            roi = (
                (float(total_profit_loss) / float(total_amount_bet)) * 100
                if total_amount_bet > 0
                else 0.0
            )

            # Calculate the Sharpe ratio
            returns = []
            for bet in bets:
                if bet["result"] == "win":
                    returns.append(bet["win_amount"] / bet["amount"])
                elif bet["result"] == "loss":
                    returns.append(-1)
                elif bet["result"] == "push":
                    returns.append(0)
            risk_free_rate = 0
            if len(returns) > 1:
                mean_return = np.mean(returns) - risk_free_rate
                std_return = np.std(returns, ddof=1)
                sharpe_ratio = (
                    mean_return / std_return if std_return != 0 else float("inf")
                )
            else:
                sharpe_ratio = float("nan")

            # Calculate the Sortino ratio
            downside_returns = [r for r in returns if r < 0]
            if len(downside_returns) > 1:
                downside_deviation = np.std(downside_returns, ddof=1)
            elif len(downside_returns) == 1:
                downside_deviation = 0.0
            else:
                downside_deviation = float("nan")
            if (
                downside_deviation
                and not np.isnan(downside_deviation)
                and downside_deviation != 0
            ):
                sortino_ratio = mean_return / downside_deviation
            else:
                sortino_ratio = float("nan")

            # Print the results
            print(f"\nModel performance against {line} lines:")
            print(f"Win rate: {win_rate:.2f}%")
            print(f"W-L-P: {wins}-{losses}-{pushes}")
            print(f"Total amount bet: {total_amount_bet}")
            print(f"Total profit: ${total_profit_loss:.2f}")
            print(f"ROI: {roi:.2f}%")
            print(f"Sharpe ratio: {sharpe_ratio:.2f}")
            print(f"Sortino ratio: {sortino_ratio:.2f}")


if __name__ == "__main__":
    main()
