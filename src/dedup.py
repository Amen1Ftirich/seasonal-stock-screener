from __future__ import annotations

import pandas as pd


def deduplicate_opportunities(
    opportunities: pd.DataFrame,
    entry_tolerance_days: int = 3,
    holding_tolerance_days: int = 5,
) -> pd.DataFrame:
    """
    Remove near-duplicate seasonal windows for the same ticker.

    Keeps the highest-ranked candidate and suppresses candidates
    with very similar entry dates and holding periods.
    """

    if opportunities.empty:
        return opportunities.copy()

    data = opportunities.copy()

    data["Upcoming Entry Date"] = pd.to_datetime(
        data["Upcoming Entry Date"]
    )

    data = data.sort_values("Rank")

    selected_rows = []

    for _, candidate in data.iterrows():

        duplicate = False

        for selected in selected_rows:

            if candidate["Ticker"] != selected["Ticker"]:
                continue

            entry_difference = abs(
                (
                    candidate["Upcoming Entry Date"]
                    - selected["Upcoming Entry Date"]
                ).days
            )

            holding_difference = abs(
                int(candidate["Holding Days"])
                - int(selected["Holding Days"])
            )

            if (
                entry_difference <= entry_tolerance_days
                and holding_difference <= holding_tolerance_days
            ):
                duplicate = True
                break

        if not duplicate:
            selected_rows.append(candidate)

    result = pd.DataFrame(selected_rows)

    if result.empty:
        return result

    result = result.reset_index(drop=True)

    result["Rank"] = range(
        1,
        len(result) + 1,
    )

    return result