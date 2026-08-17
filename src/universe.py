from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from src.config import CACHE_DIR


SP500_CACHE = CACHE_DIR / "sp500_universe.csv"

SP500_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_S%26P_500_companies"
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