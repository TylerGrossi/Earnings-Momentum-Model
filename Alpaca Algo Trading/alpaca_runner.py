# =============================================================================
# Earnings Momentum → Alpaca — single file: signals, orders, and 3:55 PM ET schedule
# =============================================================================
# Usage:
#   python alpaca_runner.py              # run once now (dry run)
#   python alpaca_runner.py --live       # run once now, submit orders
#   python alpaca_runner.py --schedule   # run at 3:55 PM ET daily (dry run)
#   python alpaca_runner.py --schedule --live   # run at 3:55 PM ET daily, submit orders
# =============================================================================

import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Path to returns_tracker and .streamlit/secrets.toml (project root)
PARENT_DIR = Path(__file__).resolve().parent.parent
SECRETS_FILE = PARENT_DIR / ".streamlit" / "secrets.toml"

# Load config from .streamlit/secrets.toml
def _load_config():
    api_key = ""
    secret_key = ""
    paper = True
    if SECRETS_FILE.exists():
        try:
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib
            with open(SECRETS_FILE, "rb") as f:
                s = tomllib.load(f)
            api_key = str(s.get("ALPACA_API_KEY", "")).strip()
            secret_key = str(s.get("ALPACA_SECRET_KEY", "")).strip()
            paper = s.get("PAPER", True)
            if isinstance(paper, str):
                paper = paper.strip().lower() in ("true", "1", "yes")
        except Exception:
            pass
    return api_key, secret_key, paper

ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER = _load_config()

MARKET_TZ = ZoneInfo("America/New_York")
RUN_AT_HOUR = 15   # 3 PM
RUN_AT_MINUTE = 55

RETURNS_FILE = PARENT_DIR / "returns_tracker.csv"

# Position sizing
POSITION_FRACTION = 0.05
FIXED_DOLLARS = None  # e.g. 1000 or None


def validate_config():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise ValueError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in "
            ".streamlit/secrets.toml (from https://app.alpaca.markets)."
        )


# -----------------------------------------------------------------------------
# Signals from returns_tracker
# -----------------------------------------------------------------------------

def _entry_date(earnings_date, timing):
    """BMO = day before earnings, AMC = day of earnings."""
    if earnings_date is None or pd.isna(earnings_date):
        return None
    d = pd.to_datetime(earnings_date).date() if hasattr(earnings_date, "date") else pd.Timestamp(earnings_date).date()
    timing_upper = (str(timing).strip().upper() if pd.notna(timing) and timing else "") or "AMC"
    if "BMO" in timing_upper:
        return d - timedelta(days=1)
    return d


def get_todays_buy_list():
    """Tickers that should be bought today (entry day); run before 4pm ET."""
    if not RETURNS_FILE.exists():
        return []

    df = pd.read_csv(RETURNS_FILE)
    if df.empty or "Ticker" not in df.columns or "Earnings Date" not in df.columns:
        return []

    df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors="coerce")
    today = datetime.now(MARKET_TZ).date()

    if "Date Check" in df.columns:
        df = df[df["Date Check"].fillna("").str.strip() != "DATE PASSED"].copy()
    df["Earnings Timing"] = df.get("Earnings Timing", "AMC")

    df["_entry_date"] = df.apply(
        lambda r: _entry_date(r["Earnings Date"], r.get("Earnings Timing")),
        axis=1,
    )
    df = df[df["_entry_date"].notna()].copy()
    today_entries = df[df["_entry_date"] == today].copy()

    now_et = datetime.now(MARKET_TZ)
    def earnings_not_yet_passed(row):
        ed = row["Earnings Date"]
        if pd.isna(ed):
            return True
        ed_date = ed.date() if hasattr(ed, "date") else pd.Timestamp(ed).date()
        if ed_date > today:
            return True
        if ed_date < today:
            return False
        timing = str(row.get("Earnings Timing", "")).strip().upper()
        if "BMO" in timing:
            return False
        return now_et.hour < 16

    today_entries = today_entries[today_entries.apply(earnings_not_yet_passed, axis=1)]

    return [
        {
            "ticker": str(r["Ticker"]).strip().upper(),
            "earnings_date": r["Earnings Date"],
            "timing": str(r.get("Earnings Timing", "AMC")).strip() or "AMC",
            "price": r.get("Price"),
            "fiscal_quarter": r.get("Fiscal Quarter"),
        }
        for _, r in today_entries.iterrows()
    ]


# -----------------------------------------------------------------------------
# Alpaca orders
# -----------------------------------------------------------------------------

def run_trading(dry_run: bool):
    """Get signals and place MOC orders (or dry run)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    validate_config()
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
    signals = get_todays_buy_list()

    if not signals:
        print("No buy signals for today. Ensure returns_tracker.csv is up to date and today is an entry day.")
        return

    equity = float(client.get_account().equity or 0)
    per_position = FIXED_DOLLARS if FIXED_DOLLARS else equity * POSITION_FRACTION

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"[{datetime.now(MARKET_TZ).strftime('%Y-%m-%d %H:%M')} ET] {mode}")
    print(f"Account equity: ${equity:,.2f}  |  Per position: ${per_position:,.2f}")
    print(f"Signals: {len(signals)}")
    print()

    for s in signals:
        ticker = s["ticker"]
        price = s.get("price")
        if price is not None and not (isinstance(price, str) and (not price or price == "nan")):
            try:
                p = float(price)
            except (TypeError, ValueError):
                p = None
        else:
            p = None

        qty = int(per_position / p) if p and p > 0 else 1
        if qty < 1:
            print(f"  Skip {ticker}: position size would be 0 shares (price ${p:.2f})")
            continue

        if dry_run:
            print(f"  Would order: BUY {qty} {ticker} @ MOC (4pm ET)")
            continue

        try:
            order_request = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.CLS,
            )
            order = client.submit_order(order_data=order_request)
            print(f"  Submitted: BUY {qty} {ticker} @ MOC -> order id {order.id}")
        except Exception as e:
            print(f"  Error {ticker}: {e}")

    if dry_run:
        print("\nUse --live to submit orders to Alpaca.")


# -----------------------------------------------------------------------------
# Scheduler: run at 3:55 PM ET every day
# -----------------------------------------------------------------------------

def next_355_et():
    """Next 3:55 PM ET as naive datetime for comparison."""
    now = datetime.now(MARKET_TZ)
    today = now.replace(hour=RUN_AT_HOUR, minute=RUN_AT_MINUTE, second=0, microsecond=0)
    if now >= today:
        today += timedelta(days=1)
    return today


def run_scheduler(live: bool):
    """Loop: wait until 3:55 PM ET, run trading, repeat."""
    dry_run = not live
    print(f"Scheduler started. Will run at {RUN_AT_HOUR}:{RUN_AT_MINUTE:02d} PM ET daily ({'LIVE' if live else 'DRY RUN'}).")
    print("Leave this window open. Press Ctrl+C to stop.")
    print()

    while True:
        target = next_355_et()
        now = datetime.now(MARKET_TZ)
        wait_sec = (target - now).total_seconds()
        if wait_sec <= 0:
            wait_sec = 60

        # Sleep in chunks so we don't oversleep by much
        end_wait = time.time() + wait_sec
        while time.time() < end_wait:
            time.sleep(min(60, max(1, end_wait - time.time())))
            if time.time() >= end_wait:
                break
        # Ensure we're at or past 3:55 PM ET
        while datetime.now(MARKET_TZ) < target:
            time.sleep(5)

        # Run at 3:55 PM ET
        print(f"\n--- Scheduled run at {datetime.now(MARKET_TZ).strftime('%Y-%m-%d %H:%M')} ET ---")
        try:
            run_trading(dry_run=dry_run)
        except Exception as e:
            print(f"Scheduled run error: {e}")
        print("--- Next run at 3:55 PM ET tomorrow ---\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    argv = [a.lower() for a in sys.argv[1:]]
    live = "--live" in argv
    schedule = "--schedule" in argv

    if schedule:
        run_scheduler(live=live)
    else:
        if live:
            print("LIVE mode: orders will be submitted to Alpaca.")
        run_trading(dry_run=not live)


if __name__ == "__main__":
    main()
