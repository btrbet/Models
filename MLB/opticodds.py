import requests
import os
from dotenv import load_dotenv


def get_fixture_id(date: str, player_name: str) -> str:
    """
    Find the fixture for a given date and team.
    """

    load_dotenv()
    api_key = os.environ.get("OPTICODDS_API_KEY")

    url = f"https://api.opticodds.com/api/v3/fixtures?sport=Baseball&league=MLB&start_date={date}"

    headers = {
        "accept": "application/json",
        "X-Api-Key": api_key,
    }

    response = requests.get(url, headers=headers)

    # Search through the 'data' key in the response for the 'id' of the fixture that contains the
    # player_name as the value of the 'home_starter' or 'away_starter' key
    fixture_id = ""
    for fixture in response.json()["data"]:
        if (
            player_name.lower() in fixture["home_starter"].lower()
            or player_name.lower() in fixture["away_starter"].lower()
        ):
            fixture_id = fixture["id"]
            break

    return fixture_id


def fetch_odds(fixture_id: str, player_name: str) -> dict:
    """
    Fetch the odds for a given fixture ID.
    """
    load_dotenv()
    api_key = os.environ.get("OPTICODDS_API_KEY")

    # books = ["draftkings", "fanduel", "betmgm", "espn", "caesars"]
    books = ["draftkings"]

    url = f"https://api.opticodds.com/api/v3/fixtures/odds/historical?fixture_id={fixture_id}&market=player_strikeouts&sportsbook=draftkings&is_main=true"
    headers = {
        "accept": "application/json",
        "X-Api-Key": api_key,
    }

    response = requests.get(url, headers=headers)

    # Search through the ['data']['odds'] key in the response for the 'points' and 'price' of
    # the 'clv' key where the value of the 'sportsbook' key is 'draftkings' and the value of the
    # 'selection' key is the player_name
    ret_val = {
        "draftkings": {
            "over_points": 0,
            "over_price": 0,
            "under_points": 0,
            "under_price": 0,
        }
    }
    data = response.json().get("data", [])
    for fixture in data:
        for odds in fixture.get("odds", []):
            if odds.get("selection").lower() == player_name.lower():
                # For both the selection_line=over and selection_line=under, get the 'points' and 'price' values of the 'clv' key
                for selection_line in ["over", "under"]:
                    clv = odds.get("clv", {})
                    if clv not in [None, {}]:
                        ret_val["draftkings"][f"{selection_line}_price"] = clv.get(
                            "price", 0
                        )
                        ret_val["draftkings"][f"{selection_line}_points"] = clv.get(
                            "points", 0
                        )

                return ret_val  # Return as soon as found

    return ret_val
