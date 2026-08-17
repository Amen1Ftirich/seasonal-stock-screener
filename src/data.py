from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import PRICE_CACHE_DIR


def get_price_cache_path(ticker: str) -> Path:
    ticker = ticker.upper().replace(".", "-")
    return PRICE_CACHE_DIR / f"{ticker}.parquet"


def download_price_history(
    ticker: str,
    start: str = "2000-01-01",
    end: str | None = None,
) -> pd.DataFrame:

    ticker = ticker.upper()

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No price data returned for {ticker}")

    # yfinance can sometimes return MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    required_columns = {
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"{ticker} data is missing columns: {sorted(missing)}"
        )

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    return data


def save_price_cache(ticker: str, data: pd.DataFrame) -> None:
    path = get_price_cache_path(ticker)
    data.to_parquet(path, index=False)


def load_price_cache(ticker: str) -> pd.DataFrame | None:
    path = get_price_cache_path(ticker)

    if not path.exists():
        return None

    data = pd.read_parquet(path)
    data["Date"] = pd.to_datetime(data["Date"])

    return data


def get_price_history(
    ticker: str,
    start: str = "2000-01-01",
    force_refresh: bool = False,
) -> pd.DataFrame:

    if not force_refresh:
        cached = load_price_cache(ticker)

        if cached is not None:
            return cached

    data = download_price_history(
        ticker=ticker,
        start=start,
    )

    save_price_cache(ticker, data)

    return data
def load_price_map(
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load historical prices for multiple tickers once.

    Returns:
        price_map:
            {
                "AAPL": DataFrame,
                "MSFT": DataFrame,
                ...
            }

        errors:
            DataFrame describing tickers that failed.
    """

    price_map = {}
    errors = []

    cleaned_tickers = sorted(
        {
            ticker.upper().strip()
            for ticker in tickers
            if ticker.strip()
        }
    )

    for ticker in cleaned_tickers:

        try:
            price_map[ticker] = get_price_history(
                ticker
            )

        except Exception as exc:
            errors.append(
                {
                    "Ticker": ticker,
                    "Error": str(exc),
                }
            )

    return (
        price_map,
        pd.DataFrame(errors),
    )