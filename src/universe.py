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
SP500_HISTORY_URL = (
    "https://en.wikipedia.org/wiki/"
    "Historical_components_of_the_S%26P_500"
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

def _clean_sp500_changes(
    changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize parsed constituent changes.
    """

    result = changes.copy()

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    result["Added"] = (
        result["Added"]
        .apply(_normalize_ticker)
    )

    result["Removed"] = (
        result["Removed"]
        .apply(_normalize_ticker)
    )

    result = (
        result
        .dropna(
            subset=["Date"]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return result

def download_sp500_changes() -> pd.DataFrame:
    """
    Download historical S&P 500 constituent changes.

    Source:
        Wikipedia historical S&P 500 components page.

    Expected table structure:

        Effective Date
        Added -> Ticker
        Added -> Security
        Removed -> Ticker
        Removed -> Security
        Reason
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
        SP500_HISTORY_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text)
    )

    for table in tables:

        #
        # The historical table should have:
        #
        # Effective Date
        # Added Ticker
        # Added Security
        # Removed Ticker
        # Removed Security
        #
        # pandas will usually parse this as
        # a MultiIndex.
        #

        if isinstance(
            table.columns,
            pd.MultiIndex,
        ):

            date_column = None
            added_ticker_column = None
            removed_ticker_column = None

            for column in table.columns:

                parts = [
                    str(part)
                    .strip()
                    .lower()

                    for part in column
                ]

                joined = " ".join(parts)

                if (
                    "effective date" in joined
                    or (
                        "date" in joined
                        and date_column is None
                    )
                ):
                    date_column = column

                if (
                    "added" in joined
                    and "ticker" in joined
                ):
                    added_ticker_column = column

                if (
                    "removed" in joined
                    and "ticker" in joined
                ):
                    removed_ticker_column = column

            if (
                date_column is not None
                and added_ticker_column is not None
                and removed_ticker_column is not None
            ):

                changes = pd.DataFrame(
                    {
                        "Date":
                            table[
                                date_column
                            ],

                        "Added":
                            table[
                                added_ticker_column
                            ],

                        "Removed":
                            table[
                                removed_ticker_column
                            ],
                    }
                )

                cleaned = _clean_sp500_changes(
                    changes
                )

                if not cleaned.empty:
                    return cleaned

        #
        # Fallback in case pandas gives us
        # ordinary one-level columns.
        #

        column_names = [
            str(column)
            .strip()
            .lower()

            for column in table.columns
        ]

        if not any(
            "date" in name
            for name in column_names
        ):
            continue

        #
        # Historical table has at least:
        #
        # Date
        # Added ticker
        # Added security
        # Removed ticker
        # Removed security
        #
        if table.shape[1] < 5:
            continue

        first_column = pd.to_datetime(
            table.iloc[:, 0],
            errors="coerce",
        )

        valid_date_ratio = (
            first_column.notna().mean()
        )

        if valid_date_ratio < 0.50:
            continue

        changes = pd.DataFrame(
            {
                "Date":
                    table.iloc[:, 0],

                "Added":
                    table.iloc[:, 1],

                "Removed":
                    table.iloc[:, 3],
            }
        )

        cleaned = _clean_sp500_changes(
            changes
        )

        if not cleaned.empty:
            return cleaned

    raise ValueError(
        "Could not locate the historical "
        "S&P 500 component-change table."
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

    Includes current members plus historical
    additions and removals.
    """

    start = pd.Timestamp(
        start_date
    )

    today = pd.Timestamp(
        date.today()
    )

    #
    # Normalize current members too, just to make
    # absolutely sure this set contains strings only.
    #

    tickers = set()

    for ticker in get_sp500_tickers():

        normalized = _normalize_ticker(
            ticker
        )

        if normalized is not None:

            tickers.add(
                normalized
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

    #
    # Historical additions
    #

    for ticker in relevant["Added"]:

        normalized = _normalize_ticker(
            ticker
        )

        if normalized is not None:

            tickers.add(
                normalized
            )

    #
    # Historical removals
    #

    for ticker in relevant["Removed"]:

        normalized = _normalize_ticker(
            ticker
        )

        if normalized is not None:

            tickers.add(
                normalized
            )

    #
    # Final defensive check:
    # return strings only.
    #

    tickers = {
        ticker
        for ticker in tickers
        if isinstance(ticker, str)
        and ticker.strip()
    }

    return sorted(
        tickers
    )

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