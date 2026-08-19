from src.sec_resolver import (
    resolve_sp500_ciks,
)


print(
    "Resolving historical S&P companies "
    "to permanent SEC CIKs..."
)


result = (
    resolve_sp500_ciks()
)


result.to_csv(
    "data/cache/sec_identifier_resolution.csv",
    index=False,
)


resolved = (
    result[
        "CIK"
    ].notna()
)


print()
print(
    "======================================"
)

print(
    "HISTORICAL SEC CIK RESOLUTION"
)

print(
    "======================================"
)


print(
    f"Ticker/name records: "
    f"{len(result)}"
)


print(
    f"Resolved: "
    f"{resolved.sum()}"
)


print(
    f"Unresolved: "
    f"{(~resolved).sum()}"
)


print(
    f"Coverage: "
    f"{resolved.mean():.2%}"
)


print()
print(
    "Resolution methods:"
)


print(
    result[
        "Resolution"
    ]
    .value_counts()
    .to_string()
)


print()
print(
    "Previously suspicious tickers:"
)


check = result[
    result[
        "Ticker"
    ].isin(
        [
            "AEP",
            "CMA",
            "HOLX",
            "EA",
            "ANSS",
            "YHOO",
            "LEH",
            "CELG",
        ]
    )
]


print(
    check[
        [
            "Ticker",
            "CIK",
            "Resolution",
            "Input Name",
            "Matched SEC Name",
            "Score",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Unresolved:"
)


unresolved = result[
    result[
        "CIK"
    ].isna()
]


print(
    ", ".join(
        unresolved[
            "Ticker"
        ].tolist()
    )
)   