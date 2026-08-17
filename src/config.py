from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PRICE_CACHE_DIR = CACHE_DIR / "prices"

BENCHMARK = "SPY"

DEFAULT_LOOKBACK_YEARS = 15

HOLDING_PERIODS = [5, 10, 15, 20, 30]

PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)