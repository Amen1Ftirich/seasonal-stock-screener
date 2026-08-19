from __future__ import annotations

import json
import os
import time

from pathlib import Path

import requests


SEC_TICKER_URL = (
    "https://www.sec.gov/files/"
    "company_tickers.json"
)

SEC_COMPANYFACTS_URL = (
    "https://data.sec.gov/api/xbrl/"
    "companyfacts/CIK{cik:010d}.json"
)


CACHE_DIR = Path(
    "data/cache/sec"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

def normalize_ticker(
    ticker: str,
) -> str:
    """
    Normalize ticker formatting across sources.

    Examples:
        BRK.B  -> BRK-B
        ALLE | -> ALLE
    """

    value = (
        str(ticker)
        .replace("\xa0", " ")
        .strip()
        .upper()
    )

    if "|" in value:
        value = (
            value
            .split("|", 1)[0]
            .strip()
        )

    return value.replace(".", "-")

def get_sec_headers() -> dict[str, str]:
    """
    SEC automated requests should identify
    the application/user in User-Agent.
    """

    user_agent = os.environ.get(
        "SEC_USER_AGENT"
    )

    if not user_agent:

        raise RuntimeError(
            "SEC_USER_AGENT is not set.\n"
            "In PowerShell run something like:\n\n"
            '$env:SEC_USER_AGENT='
            '"Seasonal Stock Screener '
            'your-email@example.com"'
        )

    return {
        "User-Agent":
            user_agent,

        "Accept-Encoding":
            "gzip, deflate",

        "Host":
            "www.sec.gov",
    }


def _download_json(
    url: str,
    host: str,
) -> dict:

    headers = get_sec_headers()

    headers["Host"] = host

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    #
    # Conservative request pacing.
    #

    time.sleep(
        0.20
    )

    return response.json()


def load_sec_ticker_map(
    refresh: bool = False,
) -> dict[str, dict]:
    """
    Return:

        normalized ticker ->
        {
            cik,
            ticker,
            title
        }
    """

    cache_file = (
        CACHE_DIR
        / "company_tickers.json"
    )


    if (
        cache_file.exists()
        and not refresh
    ):

        with cache_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            raw = json.load(
                file
            )

    else:

        raw = _download_json(
            SEC_TICKER_URL,
            host="www.sec.gov",
        )

        with cache_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                raw,
                file,
            )


    ticker_map = {}


    for record in raw.values():

        ticker = normalize_ticker(
            record["ticker"]
        )

        ticker_map[
            ticker
        ] = {
            "cik":
                int(
                    record[
                        "cik_str"
                    ]
                ),

            "ticker":
                ticker,

            "title":
                record[
                    "title"
                ],
        }


    return ticker_map


def get_cik_for_ticker(
    ticker: str,
    ticker_map: dict | None = None,
) -> int | None:

    if ticker_map is None:

        ticker_map = (
            load_sec_ticker_map()
        )


    normalized = normalize_ticker(
        ticker
    )


    record = ticker_map.get(
        normalized
    )


    if record is None:
        return None


    return int(
        record[
            "cik"
        ]
    )


def get_companyfacts_by_cik(
    cik: int,
    refresh: bool = False,
) -> dict:
    """
    Download and locally cache one company's
    full SEC Company Facts JSON.
    """

    cik = int(
        cik
    )


    cache_file = (
        CACHE_DIR
        / f"companyfacts_{cik:010d}.json"
    )


    if (
        cache_file.exists()
        and not refresh
    ):

        with cache_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )


    url = (
        SEC_COMPANYFACTS_URL.format(
            cik=cik
        )
    )


    data = _download_json(
        url,
        host="data.sec.gov",
    )


    with cache_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
        )


    return data


def get_companyfacts_for_ticker(
    ticker: str,
    refresh: bool = False,
) -> dict:

    ticker_map = (
        load_sec_ticker_map()
    )


    cik = get_cik_for_ticker(
        ticker=ticker,
        ticker_map=ticker_map,
    )


    if cik is None:

        raise KeyError(
            f"No SEC CIK mapping "
            f"found for {ticker}"
        )


    return get_companyfacts_by_cik(
        cik=cik,
        refresh=refresh,
    )