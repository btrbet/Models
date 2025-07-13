#!/usr/bin/env python3

from pybaseball import  playerid_lookup
from pybaseball import get_splits
import argparse
from scipy.stats import poisson, nbinom
import math
import mlbstatsapi
import pandas as pd
import matplotlib.pyplot as plt

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


def get_pitching_game_log_df(player_id, year):
    """
    Returns a DataFrame of a player's pitching stats for each game in a given season.

    Parameters:
    - player_id (int or str): MLBAM id of the player
    - year (int or str): The season year

    Returns:
    - df (pd.DataFrame): DataFrame where each row is a game's pitching stat line
    """
    mlb = mlbstatsapi.Mlb()
    stats = ['gameLog']
    groups = ['pitching']
    stat_dict = mlb.get_player_stats(player_id, stats=stats, groups=groups, season=str(year))
    game_log = stat_dict['pitching']['gamelog']
    rows = []
    for split in game_log.splits:
        row = split.stat.__dict__.copy()
        # Add some useful metadata if available
        row['date'] = getattr(split, 'date', None)
        row['opponent'] = getattr(split, 'opponent', None)
        row['game_id'] = getattr(split, 'game_id', None)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def plot_strikeouts_per_game(df):
    """
    Plots the number of strikeouts in each game.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least 'date' and 'strikeouts' columns
    """
    if 'date' not in df.columns or 'strikeouts' not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'strikeouts' columns.")
    
    # Convert date to datetime if not already
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate average strikeouts per game
    avg_strikeouts = calculate_average_strikeouts_per_game(df)
    avg_strikeouts_trailing_7 = calculate_average_strikeouts_per_game(df, trailing_days=7)
    avg_strikeouts_trailing_14 = calculate_average_strikeouts_per_game(df, trailing_days=14)
    avg_strikeouts_trailing_30 = calculate_average_strikeouts_per_game(df, trailing_days=30)
    avg_strikeouts_trailing_60 = calculate_average_strikeouts_per_game(df, trailing_days=60)
    avg_strikeouts_trailing_90 = calculate_average_strikeouts_per_game(df, trailing_days=90)

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['strikeouts'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Strikeouts')
    plt.title('Strikeouts By Game')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Rainbow order: red, orange, yellow, green, blue, indigo
    avg_lines = [
        ('Average (season)', avg_strikeouts, 'red'),
        ('Average (90 days)', avg_strikeouts_trailing_90, 'orange'),
        ('Average (60 days)', avg_strikeouts_trailing_60, 'yellow'),
        ('Average (30 days)', avg_strikeouts_trailing_30, 'green'),
        ('Average (14 days)', avg_strikeouts_trailing_14, 'blue'),
        ('Average (7 days)', avg_strikeouts_trailing_7, 'indigo'),
    ]
    max_date = df['date'].max()
    for i, (label, avg, color) in enumerate(avg_lines):
        if i == 0:
            # Season-long average: full width
            plt.axhline(y=avg, color=color, linestyle='--', label=f'{label}: {avg:.2f}')
        else:
            # Trailing averages: horizontal lines from start_date to max_date
            days = [90, 60, 30, 14, 7][i-1]
            start_date = max_date - pd.Timedelta(days=days)
            plt.hlines(y=avg, xmin=start_date, xmax=max_date, colors=color, linestyles='--', label=f'{label}: {avg:.2f}')

    plt.legend()
    plt.show()


def calculate_average_strikeouts_per_game(df, trailing_days=None):
    """
    Calculates the average strikeouts per game from a DataFrame, optionally for the trailing N days.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least 'gamesplayed' and 'strikeouts' columns
    - trailing_days (int or None): If provided, only include games from the most recent N days

    Returns:
    - float: Average strikeouts per game
    """
    if 'gamesplayed' not in df.columns or 'strikeouts' not in df.columns:
        raise ValueError("DataFrame must contain 'gamesplayed' and 'strikeouts' columns.")
    filtered_df = df
    if trailing_days is not None:
        if 'date' not in df.columns:
            raise ValueError("DataFrame must contain 'date' column to filter by trailing_days.")
        filtered_df = df.copy()
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        max_date = filtered_df['date'].max()
        min_date = max_date - pd.Timedelta(days=trailing_days)
        filtered_df = filtered_df[filtered_df['date'] >= min_date]
    total_strikeouts = filtered_df['strikeouts'].sum()
    total_games = filtered_df['gamesplayed'].sum()
    if total_games == 0:
        return 0.0
    return total_strikeouts / total_games


def main():
    YEAR = 2025
    parser = argparse.ArgumentParser(description="Get pitching stats for a player")
    parser.add_argument("first", help="First name of the player")
    parser.add_argument("last", help="Last name of the player")
    parser.add_argument("--line", help="The current K/G line hung for the player", type=float)
    args = parser.parse_args()

    player = find_player(args.first, args.last)
    print(player)

    df = get_pitching_game_log_df(player['key_mlbam'][0], YEAR)
    print(df.columns)
    print(df.head())

    plot_strikeouts_per_game(df)

if __name__ == "__main__":
    main()