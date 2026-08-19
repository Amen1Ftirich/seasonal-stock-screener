import pandas as pd

from src.sec_data import (
    load_sec_ticker_map,
    normalize_ticker,
)

from src.universe import (
    get_sp500_historical_union,
)


print(
    "Loading historical S&P 500 union..."
)


historical_union = sorted(
    get_sp500_historical_union(
        start_date="2006-01-01"
    )
)


print(
    f"Historical tickers: "
    f"{len(historical_union)}"
)


print(
    "Loading SEC ticker/CIK map..."
)


sec_map = (
    load_sec_ticker_map()
)


print(
    f"SEC ticker mappings: "
    f"{len(sec_map)}"
)


rows = []


for ticker in historical_union:

    normalized = (
        normalize_ticker(
            ticker
        )
    )

    record = sec_map.get(
        normalized
    )


    if record is None:

        rows.append(
            {
                "Ticker":
                    ticker,

                "Normalized":
                    normalized,

                "Mapped":
                    False,

                "CIK":
                    None,

                "SEC Name":
                    None,
            }
        )

    else:

        rows.append(
            {
                "Ticker":
                    ticker,

                "Normalized":
                    normalized,

                "Mapped":
                    True,

                "CIK":
                    record[
                        "cik"
                    ],

                "SEC Name":
                    record[
                        "title"
                    ],
            }
        )


coverage = pd.DataFrame(
    rows
)


mapped = int(
    coverage[
        "Mapped"
    ].sum()
)


total = len(
    coverage
)


missing = (
    coverage[
        ~coverage[
            "Mapped"
        ]
    ]
    .copy()
)


print()
print(
    "======================================"
)

print(
    "SEC IDENTIFIER COVERAGE"
)

print(
    "======================================"
)

print(
    f"Historical tickers: "
    f"{total}"
)

print(
    f"Mapped directly: "
    f"{mapped}"
)

print(
    f"Missing: "
    f"{len(missing)}"
)

print(
    f"Direct coverage: "
    f"{mapped / total:.2%}"
)


print()
print(
    "Missing tickers:"
)

print(
    ", ".join(
        missing[
            "Ticker"
        ].astype(str)
    )
)


coverage.to_csv(
    "data/cache/sec_ticker_coverage.csv",
    index=False,
)


missing.to_csv(
    "data/cache/sec_missing_tickers.csv",
    index=False,
)