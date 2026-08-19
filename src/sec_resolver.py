from __future__ import annotations

import re
import time

from difflib import (
    SequenceMatcher,
)

from pathlib import Path

import pandas as pd
import requests

from src.sec_data import (
    get_sec_headers,
    load_sec_ticker_map,
    normalize_ticker,
)

from src.universe import (
    get_sp500_ticker_name_map,
)


SEC_CIK_LOOKUP_URL = (
    "https://www.sec.gov/Archives/"
    "edgar/cik-lookup-data.txt"
)


CACHE_DIR = Path(
    "data/cache/sec"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CIK_LOOKUP_CACHE = (
    CACHE_DIR
    / "cik-lookup-data.txt"
)
CIK_OVERRIDES = {
    "AEP": {
        "cik": 4904,
        "name": "AMERICAN ELECTRIC POWER CO INC",
    },

    "EQR": {
        "cik": 906107,
        "name": "EQUITY RESIDENTIAL",
    },
}

def normalize_company_name(
    name: str,
) -> str:
    """
    Normalize company names for entity matching.

    Examples:

        ELECTRONIC ARTS INC.
            -> ELECTRONIC ARTS

        AMERICAN ELECTRIC POWER CO INC
            -> AMERICAN ELECTRIC POWER
    """

    value = str(
        name
    ).upper()


    value = value.replace(
        "&",
        " AND ",
    )


    value = re.sub(
        r"[^A-Z0-9 ]+",
        " ",
        value,
    )


    tokens = value.split()


    removable_suffixes = {
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "LTD",
        "LIMITED",
        "LLC",
        "PLC",
        "LP",
    }


    while (
        tokens
        and tokens[-1]
        in removable_suffixes
    ):

        tokens.pop()


    #
    # SEC names sometimes end with jurisdiction
    # annotations such as /DE/ or /NEW/.
    #

    removable_tail = {
        "DE",
        "NEW",
        "NV",
    }


    while (
        tokens
        and tokens[-1]
        in removable_tail
    ):

        tokens.pop()


    return " ".join(
        tokens
    )


def download_historical_cik_names(
    refresh: bool = False,
) -> Path:
    """
    Download SEC's historically cumulative
    company-name / CIK file.
    """

    if (
        CIK_LOOKUP_CACHE.exists()
        and not refresh
    ):

        return CIK_LOOKUP_CACHE


    headers = get_sec_headers()

    headers["Host"] = (
        "www.sec.gov"
    )


    response = requests.get(
        SEC_CIK_LOOKUP_URL,
        headers=headers,
        timeout=120,
    )

    response.raise_for_status()


    CIK_LOOKUP_CACHE.write_bytes(
        response.content
    )


    time.sleep(
        0.25
    )


    return CIK_LOOKUP_CACHE


def load_historical_cik_names(
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Return unique historical:

        SEC Name
        Normalized Name
        CIK
    """

    path = (
        download_historical_cik_names(
            refresh=refresh
        )
    )


    rows = []


    with path.open(
        "r",
        encoding="latin-1",
        errors="ignore",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue


            #
            # SEC format:
            #
            # COMPANY NAME:CIK:
            #
            # rsplit protects us if a name
            # itself contains punctuation.
            #

            parts = line.rsplit(
                ":",
                2,
            )


            if len(parts) < 2:
                continue


            company_name = (
                parts[0].strip()
            )


            cik_text = (
                parts[1].strip()
            )


            if not cik_text.isdigit():
                continue


            cik = int(
                cik_text
            )


            normalized = (
                normalize_company_name(
                    company_name
                )
            )


            if not normalized:
                continue


            rows.append(
                {
                    "SEC Name":
                        company_name,

                    "Normalized Name":
                        normalized,

                    "CIK":
                        cik,
                }
            )


    result = (
        pd.DataFrame(
            rows
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )


    return result


def build_name_index(
    sec_names: pd.DataFrame,
) -> dict[str, set[int]]:

    index = {}


    for _, row in (
        sec_names.iterrows()
    ):

        name = row[
            "Normalized Name"
        ]

        cik = int(
            row["CIK"]
        )


        index.setdefault(
            name,
            set(),
        ).add(
            cik
        )


    return index


def resolve_sp500_ciks(
) -> pd.DataFrame:
    """
    Resolve historical S&P tickers to CIKs.

    Resolution order:

        1. SEC current ticker map
        2. exact historical company-name match
        3. conservative fuzzy company-name match

    Ambiguous matches are NOT silently accepted.
    """

    ticker_names = (
        get_sp500_ticker_name_map()
    )


    ticker_map = (
        load_sec_ticker_map()
    )


    sec_names = (
        load_historical_cik_names()
    )


    name_index = (
        build_name_index(
            sec_names
        )
    )


    all_sec_names = list(
        name_index.keys()
    )


    rows = []


    for ticker, names in sorted(
        ticker_names.items()
    ):

        ticker = normalize_ticker(
            ticker
        )

        #
        # ----------------------------------
        # 0. VERIFIED SEC OVERRIDES
        # ----------------------------------
        #

        override = (
            CIK_OVERRIDES.get(
                ticker
            )
        )

        if override is not None:

            rows.append(
                {
                    "Ticker":
                        ticker,

                    "CIK":
                        int(
                            override[
                                "cik"
                            ]
                        ),

                    "Resolution":
                        "verified_override",

                    "Input Name":
                        names[0]
                        if names
                        else None,

                    "Matched SEC Name":
                        override[
                            "name"
                        ],

                    "Score":
                        1.0,
                }
            )

            continue
        #
        # ----------------------------------
        # 1. CURRENT TICKER MATCH
        # ----------------------------------
        #

        ticker_record = (
            ticker_map.get(
                ticker
            )
        )


        if ticker_record is not None:

            rows.append(
                {
                    "Ticker":
                        ticker,

                    "CIK":
                        int(
                            ticker_record[
                                "cik"
                            ]
                        ),

                    "Resolution":
                        "ticker",

                    "Input Name":
                        names[0]
                        if names
                        else None,

                    "Matched SEC Name":
                        ticker_record[
                            "title"
                        ],

                    "Score":
                        1.0,
                }
            )

            continue


        #
        # ----------------------------------
        # 2. EXACT HISTORICAL NAME MATCH
        # ----------------------------------
        #

        exact_candidates = set()


        for name in names:

            normalized = (
                normalize_company_name(
                    name
                )
            )


            exact_candidates.update(
                name_index.get(
                    normalized,
                    set(),
                )
            )


        if len(
            exact_candidates
        ) == 1:

            cik = next(
                iter(
                    exact_candidates
                )
            )


            matched_rows = (
                sec_names[
                    sec_names[
                        "CIK"
                    ] == cik
                ]
            )


            rows.append(
                {
                    "Ticker":
                        ticker,

                    "CIK":
                        cik,

                    "Resolution":
                        "name_exact",

                    "Input Name":
                        " | ".join(
                            names
                        ),

                    "Matched SEC Name":
                        matched_rows[
                            "SEC Name"
                        ].iloc[0],

                    "Score":
                        1.0,
                }
            )

            continue


        if len(
            exact_candidates
        ) > 1:

            rows.append(
                {
                    "Ticker":
                        ticker,

                    "CIK":
                        None,

                    "Resolution":
                        "ambiguous_exact",

                    "Input Name":
                        " | ".join(
                            names
                        ),

                    "Matched SEC Name":
                        None,

                    "Score":
                        None,
                }
            )

            continue


        #
        # ----------------------------------
        # 3. CONSERVATIVE FUZZY MATCH
        # ----------------------------------
        #

        best = None


        for input_name in names:

            normalized_input = (
                normalize_company_name(
                    input_name
                )
            )


            if not normalized_input:
                continue


            #
            # Reduce candidate universe using
            # the first meaningful token.
            #

            first_token = (
                normalized_input.split()[0]
            )


            candidates = [
                candidate

                for candidate
                in all_sec_names

                if (
                    candidate.startswith(
                        first_token
                    )
                    or first_token
                    in candidate.split()
                )
            ]


            for candidate in candidates:

                score = (
                    SequenceMatcher(
                        None,
                        normalized_input,
                        candidate,
                    ).ratio()
                )


                if (
                    best is None
                    or score
                    > best[
                        "score"
                    ]
                ):

                    best = {
                        "input":
                            input_name,

                        "normalized":
                            candidate,

                        "score":
                            score,
                    }


        if (
            best is not None
            and best[
                "score"
            ] >= 0.92
        ):

            candidate_ciks = (
                name_index[
                    best[
                        "normalized"
                    ]
                ]
            )


            if len(
                candidate_ciks
            ) == 1:

                cik = next(
                    iter(
                        candidate_ciks
                    )
                )


                matched = (
                    sec_names[
                        (
                            sec_names[
                                "CIK"
                            ] == cik
                        )
                        &
                        (
                            sec_names[
                                "Normalized Name"
                            ]
                            == best[
                                "normalized"
                            ]
                        )
                    ]
                )


                rows.append(
                    {
                        "Ticker":
                            ticker,

                        "CIK":
                            cik,

                        "Resolution":
                            "name_fuzzy",

                        "Input Name":
                            best[
                                "input"
                            ],

                        "Matched SEC Name":
                            matched[
                                "SEC Name"
                            ].iloc[0],

                        "Score":
                            best[
                                "score"
                            ],
                    }
                )

                continue


        rows.append(
            {
                "Ticker":
                    ticker,

                "CIK":
                    None,

                "Resolution":
                    "unresolved",

                "Input Name":
                    " | ".join(
                        names
                    ),

                "Matched SEC Name":
                    None,

                "Score":
                    (
                        best[
                            "score"
                        ]
                        if best
                        is not None
                        else None
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )