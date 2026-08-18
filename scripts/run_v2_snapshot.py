from datetime import date

from src.cross_section_v2 import (
    build_cross_section,
    select_candidates,
)

from src.data import (
    get_price_history,
    load_price_map,
)

from src.universe import (
    get_sp500_tickers,
)


tickers = get_sp500_tickers()


print(
    f"Loading {len(tickers)} stocks..."
)


price_map, errors = load_price_map(
    tickers
)


spy = get_price_history(
    "SPY"
)


#
# We are currently in August.
#
# Use September as the first clean
# forward calendar-month signal.
#

snapshot = build_cross_section(
    price_map=price_map,
    benchmark_prices=spy,

    target_month=9,

    as_of_date=date.today(),

    lookback_years=15,
    recent_years=5,
)


candidates = select_candidates(
    snapshot,

    top_n=20,

    minimum_samples=8,

    minimum_long_season_rank=0.80,

    minimum_recent_season_rank=0.60,

    minimum_momentum_rank=0.50,

    minimum_downside_rank=0.40,
)


columns = [
    "Rank",
    "Ticker",

    "Season Samples",

    "Season Mean Excess",
    "Recent Mean Excess",

    "Long Seasonal Rank",
    "Recent Seasonal Rank",

    "Momentum 12-1",
    "Momentum Rank",

    "Season Q25 Excess",
    "Downside Rank",

    "Edge Score",
]


print()
print(
    candidates[
        columns
    ].to_string(
        index=False
    )
)