from __future__ import annotations

from datetime import date

import pandas as pd


def _prepare_prices(
    prices: pd.DataFrame,
    price_column: str = "Adj Close",
) -> pd.DataFrame:
    """
    Validate and clean historical price data.
    """

    required = {"Date", price_column}
    missing = required.difference(prices.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = prices[["Date", price_column]].copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.dropna(subset=["Date", price_column])
    data = data.sort_values("Date")
    data = data.drop_duplicates(subset=["Date"])
    data = data.reset_index(drop=True)

    return data


def _target_date_for_year(
    year: int,
    month: int,
    day: int,
) -> pd.Timestamp | None:
    """
    Construct the desired calendar date for a specific year.

    Returns None for invalid dates such as February 29
    during a non-leap year.
    """

    try:
        return pd.Timestamp(date(year, month, day))
    except ValueError:
        return None


def _find_entry_index(
    prices: pd.DataFrame,
    target_date: pd.Timestamp,
    max_calendar_shift: int = 7,
) -> int | None:
    """
    Find the first trading day on or after the target calendar date.

    Example:
        Target = Saturday Aug 20
        Actual entry = Monday Aug 22

    max_calendar_shift prevents accidentally entering far away
    from the intended seasonal date.
    """

    candidates = prices[
        (prices["Date"] >= target_date)
        & (
            prices["Date"]
            <= target_date + pd.Timedelta(days=max_calendar_shift)
        )
    ]

    if candidates.empty:
        return None

    return int(candidates.index[0])


def get_seasonal_returns(
    prices: pd.DataFrame,
    entry_month: int,
    entry_day: int,
    holding_days: int,
    lookback_years: int = 15,
    price_column: str = "Adj Close",
    max_calendar_shift: int = 7,
) -> pd.DataFrame:
    """
    Calculate historical returns for the same seasonal window
    across multiple years.

    Parameters
    ----------
    prices:
        Historical daily stock-price DataFrame.

    entry_month / entry_day:
        Desired calendar entry date.

    holding_days:
        Number of trading sessions after entry before exit.

    lookback_years:
        Number of most recent valid historical observations.

    price_column:
        Price series used for return calculations.

    max_calendar_shift:
        Maximum number of calendar days allowed when moving
        from the desired date to the next valid trading session.

    Returns
    -------
    DataFrame containing one observation per historical year.
    """

    if holding_days <= 0:
        raise ValueError("holding_days must be greater than zero")

    if lookback_years <= 0:
        raise ValueError("lookback_years must be greater than zero")

    data = _prepare_prices(
        prices=prices,
        price_column=price_column,
    )

    observations = []

    first_year = int(data["Date"].dt.year.min())
    last_year = int(data["Date"].dt.year.max())

    for year in range(first_year, last_year + 1):

        target_date = _target_date_for_year(
            year=year,
            month=entry_month,
            day=entry_day,
        )

        if target_date is None:
            continue

        entry_index = _find_entry_index(
            prices=data,
            target_date=target_date,
            max_calendar_shift=max_calendar_shift,
        )

        if entry_index is None:
            continue

        exit_index = entry_index + holding_days

        if exit_index >= len(data):
            continue

        entry_row = data.iloc[entry_index]
        exit_row = data.iloc[exit_index]

        # Prevent an exit from accidentally crossing into an
        # unrelated future period because of missing data.
        if exit_row["Date"].year > year + 1:
            continue

        entry_price = float(entry_row[price_column])
        exit_price = float(exit_row[price_column])

        if entry_price <= 0:
            continue

        seasonal_return = (exit_price / entry_price) - 1

        observations.append(
            {
                "Year": year,
                "Target Date": target_date,
                "Entry Date": entry_row["Date"],
                "Exit Date": exit_row["Date"],
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return": seasonal_return,
                "Win": seasonal_return > 0,
            }
        )

    results = pd.DataFrame(observations)

    if results.empty:
        return results

    # We want the most recent N VALID seasonal observations,
    # not merely the last N calendar years.
    results = (
        results.sort_values("Year")
        .tail(lookback_years)
        .reset_index(drop=True)
    )

    return results