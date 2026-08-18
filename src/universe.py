from __future__ import annotations
from datetime import date
from io import StringIO

import pandas as pd
import requests
from src.config import CACHE_DIR


SP500_CACHE = CACHE_DIR / "sp500_universe.csv"

SP500_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_S%26P_500_companies"
)
SP500_CHANGES_CACHE = (
    CACHE_DIR / "sp500_changes.csv"
)

TECH_TEST_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "AVGO",
    "META",
    "GOOGL",
    "AMZN",
    "QCOM",
    "INTC",
]


def download_sp500_table() -> pd.DataFrame:
    """
    Download the S&P 500 constituent table.

    A browser-style User-Agent is supplied because Wikipedia
    may reject Python's default urllib request.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        )
    }

    response = requests.get(
        SP500_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text)
    )

    for table in tables:
        if "Symbol" in table.columns:
            return table

    raise ValueError(
        "Could not find S&P 500 constituent table."
    )


def get_sp500_tickers(
    force_refresh: bool = False,
) -> list[str]:
    """
    Return current S&P 500 ticker symbols.

    Uses a local CSV cache after the first successful download.
    """

    if (
        SP500_CACHE.exists()
        and not force_refresh
    ):
        cached = pd.read_csv(
            SP500_CACHE
        )

        if "Symbol" in cached.columns:
            return (
                cached["Symbol"]
                .dropna()
                .astype(str)
                .tolist()
            )

    table = download_sp500_table()

    tickers = (
        table["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(
            ".",
            "-",
            regex=False,
        )
    )

    result = pd.DataFrame(
        {
            "Symbol": tickers
        }
    )

    result.to_csv(
        SP500_CACHE,
        index=False,
    )

    return result["Symbol"].tolist()
def _normalize_ticker(
    ticker,
) -> str | None:
    """
    Normalize Wikipedia tickers into Yahoo-style symbols.

    Example:
        BRK.B -> BRK-B
    """

    if pd.isna(ticker):
        return None

    ticker = str(ticker).strip()

    if (
        not ticker
        or ticker.lower() == "nan"
    ):
        return None

    return ticker.replace(
        ".",
        "-",
    )


def _column_name(
    column,
) -> str:
    """
    Convert pandas MultiIndex HTML-table
    columns into searchable strings.
    """

    if isinstance(column, tuple):

        parts = []

        for part in column:

            text = str(part).strip()

            if (
                text
                and text.lower() != "nan"
                and not text.startswith(
                    "Unnamed"
                )
            ):
                parts.append(text)

        return " ".join(parts).lower()

    return str(column).strip().lower()


def download_sp500_changes() -> pd.DataFrame:
    """
    Download the Wikipedia S&P 500 component-change
    table.

    Output:
        Date
        Added
        Removed
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        )
    }

    response = requests.get(
        SP500_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(
            response.text
        )
    )

    for table in tables:

        columns = {
            _column_name(column):
                column

            for column in table.columns
        }

        date_column = None
        added_column = None
        removed_column = None

        for name, original in columns.items():

            if (
                "date" in name
                and date_column is None
            ):
                date_column = original

            if (
                "added" in name
                and "ticker" in name
            ):
                added_column = original

            if (
                "removed" in name
                and "ticker" in name
            ):
                removed_column = original

        if (
            date_column is not None
            and added_column is not None
            and removed_column is not None
        ):

            changes = pd.DataFrame(
                {
                    "Date":
                        table[
                            date_column
                        ],

                    "Added":
                        table[
                            added_column
                        ],

                    "Removed":
                        table[
                            removed_column
                        ],
                }
            )

            changes["Date"] = (
                pd.to_datetime(
                    changes["Date"],
                    errors="coerce",
                )
            )

            changes["Added"] = (
                changes["Added"]
                .apply(
                    _normalize_ticker
                )
            )

            changes["Removed"] = (
                changes["Removed"]
                .apply(
                    _normalize_ticker
                )
            )

            changes = (
                changes
                .dropna(
                    subset=["Date"]
                )
                .sort_values("Date")
                .reset_index(drop=True)
            )

            return changes

    raise ValueError(
        "Could not locate the S&P 500 "
        "component-change table."
    )


def get_sp500_changes(
    force_refresh: bool = False,
) -> pd.DataFrame:

    if (
        SP500_CHANGES_CACHE.exists()
        and not force_refresh
    ):

        changes = pd.read_csv(
            SP500_CHANGES_CACHE
        )

        changes["Date"] = (
            pd.to_datetime(
                changes["Date"]
            )
        )

        changes["Added"] = (
            changes["Added"]
            .apply(_normalize_ticker)
        )

        changes["Removed"] = (
            changes["Removed"]
            .apply(_normalize_ticker)
        )

        return changes

    changes = download_sp500_changes()

    changes.to_csv(
        SP500_CHANGES_CACHE,
        index=False,
    )

    return changes
def get_sp500_membership(
    as_of_date: str | date | pd.Timestamp,
) -> set[str]:
    """
    Reconstruct approximate S&P 500 membership
    as of a historical date.

    Method:

        Start with today's S&P 500.

        Walk constituent changes backward.

        Reverse every change occurring after
        the requested historical date.

    Example:

        If XYZ replaced ABC in 2022,

        asking for membership in 2020:

            remove XYZ
            restore ABC
    """

    target = pd.Timestamp(
        as_of_date
    ).normalize()

    today = pd.Timestamp(
        date.today()
    ).normalize()

    if target > today:

        raise ValueError(
            "Cannot reconstruct future "
            "S&P 500 membership."
        )

    members = set(
        get_sp500_tickers()
    )

    changes = get_sp500_changes()

    relevant = changes[
        (
            changes["Date"] > target
        )
        & (
            changes["Date"] <= today
        )
    ].sort_values(
        "Date",
        ascending=False,
    )

    for _, change in relevant.iterrows():

        added = change["Added"]
        removed = change["Removed"]

        #
        # Reverse the historical change.
        #

        if added is not None:

            members.discard(
                added
            )

        if removed is not None:

            members.add(
                removed
            )

    return members
def get_sp500_historical_union(
    start_date: str | date | pd.Timestamp,
) -> list[str]:
    """
    Return every ticker that could have been an
    S&P 500 member since start_date.

    This includes current members plus historical
    additions and removals.
    """

    start = pd.Timestamp(
        start_date
    )

    today = pd.Timestamp(
        date.today()
    )

    tickers = set(
        get_sp500_tickers()
    )

    changes = get_sp500_changes()

    relevant = changes[
        (
            changes["Date"] >= start
        )
        & (
            changes["Date"] <= today
        )
    ]

    for ticker in relevant["Added"]:

        if ticker is not None:
            tickers.add(ticker)

    for ticker in relevant["Removed"]:

        if ticker is not None:
            tickers.add(ticker)

    return sorted(tickers)
def build_sp500_membership_map(
    start_date: str,
    end_date: str,
) -> dict[pd.Period, set[str]]:
    """
    Build point-in-time S&P 500 membership for
    every calendar month.

    Membership is observed at the beginning
    of each month.
    """

    periods = pd.period_range(
        start=start_date,
        end=end_date,
        freq="M",
    )

    membership_map = {}

    for number, period in enumerate(
        periods,
        start=1,
    ):

        if number % 24 == 0:

            print(
                f"Membership reconstruction "
                f"{number}/{len(periods)}"
            )

        as_of_date = (
            period.start_time
        )

        membership_map[
            period
        ] = get_sp500_membership(
            as_of_date
        )

    return membership_map