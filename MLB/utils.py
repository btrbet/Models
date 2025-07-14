from pybaseball import playerid_lookup, get_splits
import pandas as pd
import mlbstatsapi


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
    elif prob != 0.0:
        odds = (100.00 / prob) - 100.00
    else:
        odds = 0
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
    stats = ["gameLog"]
    groups = ["pitching"]
    stat_dict = mlb.get_player_stats(
        player_id, stats=stats, groups=groups, season=str(year)
    )
    game_log = stat_dict["pitching"]["gamelog"]
    rows = []
    for split in game_log.splits:
        row = split.stat.__dict__.copy()
        # Add some useful metadata if available
        row["date"] = getattr(split, "date", None)
        row["opponent"] = getattr(split, "opponent", None)
        row["game_id"] = getattr(split, "game_id", None)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


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


def calculate_average_strikeouts_per_game(df, trailing_days=None):
    """
    Calculates the average strikeouts per game from a DataFrame, optionally for the trailing N days.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least 'gamesplayed' and 'strikeouts' columns
    - trailing_days (int or None): If provided, only include games from the most recent N days

    Returns:
    - float: Average strikeouts per game
    """
    if "gamesplayed" not in df.columns or "strikeouts" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'gamesplayed' and 'strikeouts' columns."
        )
    filtered_df = df
    if trailing_days is not None:
        if "date" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'date' column to filter by trailing_days."
            )
        filtered_df = df.copy()
        filtered_df["date"] = pd.to_datetime(filtered_df["date"])
        max_date = filtered_df["date"].max()
        min_date = max_date - pd.Timedelta(days=trailing_days)
        filtered_df = filtered_df[filtered_df["date"] >= min_date]
    total_strikeouts = filtered_df["strikeouts"].sum()
    total_games = filtered_df["gamesplayed"].sum()
    if total_games == 0:
        return 0.0
    return total_strikeouts / total_games
