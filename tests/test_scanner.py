from src.scanner import scan_tickers


def test_scanner_module_imports():
    assert callable(scan_tickers)