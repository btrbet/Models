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
    Fetch the odds for a given fixture ID from multiple sportsbooks.
    """
    load_dotenv()
    api_key = os.environ.get("OPTICODDS_API_KEY")

    books = ["draftkings", "fanduel", "betmgm", "caesars"]

    url = f"https://api.opticodds.com/api/v3/fixtures/odds/historical?fixture_id={fixture_id}&market=player_strikeouts&is_main=true"
    for book in books:
        url += f"&sportsbook={book}"

    headers = {
        "accept": "application/json",
        "X-Api-Key": api_key,
    }

    response = requests.get(url, headers=headers)

    # Prepare return value for all books
    ret_val = {
        book: {
            "over_points": 0,
            "over_price": 0,
            "under_points": 0,
            "under_price": 0,
        }
        for book in books
    }
    data = response.json().get("data", [])
    for fixture in data:
        for odds in fixture.get("odds", []):
            sportsbook = odds.get("sportsbook", "").lower()
            if (
                sportsbook in books
                and odds.get("selection", "").lower() == player_name.lower()
            ):
                if odds.get("selection_line") == "over":
                    clv = odds.get("clv", {})
                    if clv not in [None, {}]:
                        ret_val[sportsbook]["over_points"] = clv.get("points", 0)
                        ret_val[sportsbook]["over_price"] = clv.get("price", 0)
                elif odds.get("selection_line") == "under":
                    clv = odds.get("clv", {})
                    if clv not in [None, {}]:
                        ret_val[sportsbook]["under_points"] = clv.get("points", 0)
                        ret_val[sportsbook]["under_price"] = clv.get("price", 0)

    return ret_val
