# ================================================
# Earnings Momentum Quant Model - with Barchart Filter
# ================================================
# UPDATES v17:
#   - Removed earnings_universe.csv - all data goes to returns_tracker.csv
#   - Simplified workflow: build_universe -> update_returns -> export_hourly_prices
#   - Cleaner code organization
# ================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time, re, os

# ------------------------------------
# CONFIG
# ------------------------------------

# Output directory
#OUTPUT_DIR = r"C:\Users\Owner\Desktop\Tyler\OneDrive\Projects\Quantitative Finance\Earnings-Momentum-Model"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

RETURNS_FILE  = os.path.join(OUTPUT_DIR, "returns_tracker.csv")
PRICES_FILE   = os.path.join(OUTPUT_DIR, "hourly_prices.csv")
         
FINVIZ_SCREENER = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/json",
}


# ------------------------------------
# FINVIZ / BARCHART SCRAPERS
# ------------------------------------

def get_all_tickers():
    """Get all tickers from Finviz screener."""
    tickers, offset = [], 0
    while True:
        url = f"{FINVIZ_SCREENER}&r={offset + 1}"
        soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, "html.parser")
        new = [cols[1].text.strip() for row in soup.select("table tr") 
               for cols in [row.find_all("td")] if len(cols) > 1 
               and cols[1].text.strip().isupper() and cols[1].text.strip().isalpha()]
        if not new: break
        tickers.extend(t for t in new if t not in tickers)
        offset += 20
    return tickers


def get_finviz_stats(ticker):
    """Get stock stats from Finviz."""
    try:
        soup = BeautifulSoup(requests.get(f"https://finviz.com/quote.ashx?t={ticker}", headers=HEADERS, timeout=10).text, "html.parser")
        name, sector = get_company_name(ticker)
        cells = soup.find_all("table")[8].find_all("td")
        data = {cells[i].get_text(strip=True): cells[i+1].get_text(strip=True) for i in range(0, len(cells), 2)}
        return {"Ticker": ticker, "Company Name": name, "Sector": sector, "Price": data.get("Price"),
                "EPS (TTM)": data.get("EPS (ttm)"), "P/E": data.get("P/E"), "Forward P/E": data.get("Forward P/E"),
                "Market Cap": data.get("Market Cap"), "Beta": data.get("Beta")}
    except:
        return {"Ticker": ticker, "Company Name": ticker, "Sector": "N/A"}


def has_buy_signal(ticker):
    """Check if ticker has Barchart buy signal."""
    try:
        soup = BeautifulSoup(requests.get(f"https://www.barchart.com/stocks/quotes/{ticker}/opinion", headers=HEADERS, timeout=10).text, "html.parser")
        sig = soup.find("span", class_="opinion-signal buy")
        return bool(sig and "Buy" in sig.text)
    except:
        return False


def get_company_name(ticker):
    """Get company name and sector from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker, info.get('sector', 'N/A')
    except:
        return ticker, 'N/A'


# ------------------------------------
# EARNINGS DATE FUNCTIONS
# ------------------------------------

def get_yfinance_earnings_date(ticker):
    """Get the most recent/upcoming earnings date from yfinance."""
    try:
        earnings_df = yf.Ticker(ticker).get_earnings_dates(limit=10)
        if earnings_df is None or earnings_df.empty:
            return None, None
        
        today = datetime.today().date()
        best_date, best_timing = None, None
        min_diff = 999
        
        for idx in earnings_df.index:
            try:
                idx_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                diff = abs((idx_date - today).days)
                if diff < min_diff and diff <= 60:
                    min_diff = diff
                    best_date = datetime.combine(idx_date, datetime.min.time())
                    if hasattr(idx, 'hour'):
                        best_timing = "BMO" if idx.hour < 12 else "AMC"
            except:
                continue
        
        return best_date, best_timing
    except:
        return None, None


def check_date_status(earnings_date, yfinance_date):
    """Check if earnings date has passed based on yfinance data."""
    try:
        if earnings_date is None or yfinance_date is None:
            return "OK"
        
        ed = safe_dt(earnings_date)
        yf_d = safe_dt(yfinance_date)
        
        if pd.isna(ed) or pd.isna(yf_d):
            return "OK"
        
        ed_date = ed.date()
        yf_date = yf_d.date()
        
        date_diff = abs((ed_date - yf_date).days)
        if date_diff > 14:
            return "OK"
        
        if yf_date < ed_date:
            return "DATE PASSED"
        
        return "OK"
    except:
        return "OK"


# ------------------------------------
# EPS DATA FUNCTIONS
# ------------------------------------

def get_eps_from_yfinance(ticker, target_date):
    """Get EPS data from yfinance."""
    result = {"EPS Estimate": None, "Reported EPS": None, "EPS Surprise (%)": None, "Earnings Timing": None}
    
    try:
        earnings_df = yf.Ticker(ticker).get_earnings_dates(limit=20)
        if earnings_df is None or earnings_df.empty:
            return result
        
        target = target_date.date() if hasattr(target_date, 'date') else pd.to_datetime(target_date).date()
        
        best_match, min_diff = None, 999
        for idx in earnings_df.index:
            try:
                idx_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                diff = abs((idx_date - target).days)
                if diff <= 2 and diff < min_diff:
                    min_diff, best_match = diff, idx
            except:
                continue
        
        if best_match is not None:
            row = earnings_df.loc[best_match]
            if 'EPS Estimate' in row.index and pd.notna(row['EPS Estimate']):
                result["EPS Estimate"] = float(row['EPS Estimate'])
            if 'Reported EPS' in row.index and pd.notna(row['Reported EPS']):
                result["Reported EPS"] = float(row['Reported EPS'])
            surp_col = 'Surprise(%)' if 'Surprise(%)' in row.index else 'Surprise (%)'
            if surp_col in row.index and pd.notna(row[surp_col]):
                result["EPS Surprise (%)"] = float(row[surp_col])
            if hasattr(best_match, 'hour'):
                result["Earnings Timing"] = "BMO" if best_match.hour < 12 else "AMC"
    except:
        pass
    
    return result


def scrape_yahoo_calendar(date):
    """Scrape Yahoo calendar for EPS data (fallback)."""
    results = {}
    try:
        url = f"https://finance.yahoo.com/calendar/earnings?day={date.strftime('%Y-%m-%d')}"
        soup = BeautifulSoup(requests.get(url, headers=HEADERS, timeout=15).text, "html.parser")
        table = soup.find("table")
        if not table: return results
        
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 5: continue
            try:
                sym = cells[0].get_text(strip=True).upper()
                timing = "BMO" if 'before' in cells[2].get_text(strip=True).lower() else ("AMC" if 'after' in cells[2].get_text(strip=True).lower() else None)
                est = cells[3].get_text(strip=True)
                est = float(est.replace(",", "")) if est and est not in ["-", "N/A", ""] else None
                act = cells[4].get_text(strip=True)
                act = float(act.replace(",", "")) if act and act not in ["-", "N/A", ""] else None
                surp = cells[5].get_text(strip=True) if len(cells) > 5 else ""
                surp = float(surp.replace("%", "").replace("+", "").replace(",", "")) if surp and surp not in ["-", "N/A", ""] else None
                results[sym] = {"EPS Estimate": est, "Reported EPS": act, "EPS Surprise (%)": surp, "Earnings Timing": timing}
            except:
                continue
    except:
        pass
    return results


# ------------------------------------
# FISCAL QUARTER
# ------------------------------------

def get_fiscal_quarter(ticker, earnings_date):
    """Get fiscal quarter from yfinance earnings_history."""
    try:
        yf_ticker = yf.Ticker(ticker)
        target = earnings_date.date() if hasattr(earnings_date, 'date') else pd.to_datetime(earnings_date).date()
        
        try:
            hist = yf_ticker.earnings_history
            if hist is not None and not hist.empty:
                best_quarter_end = None
                min_diff = 999
                
                for q_end in hist.index:
                    q_end_date = q_end.date() if hasattr(q_end, 'date') else pd.to_datetime(q_end).date()
                    days_after = (target - q_end_date).days
                    if 10 <= days_after <= 75:
                        if days_after < min_diff:
                            min_diff = days_after
                            best_quarter_end = q_end_date
                
                if best_quarter_end:
                    info = yf_ticker.info
                    fy_end_month = None
                    
                    if info.get('lastFiscalYearEnd'):
                        try:
                            fy_end_month = datetime.fromtimestamp(info['lastFiscalYearEnd']).month
                        except:
                            pass
                    
                    if fy_end_month is None:
                        q_months = sorted(set([q.month if hasattr(q, 'month') else pd.to_datetime(q).month for q in hist.index]))
                        if q_months:
                            for m in q_months:
                                next_q = (m + 3) % 12 or 12
                                if next_q not in q_months:
                                    fy_end_month = m
                                    break
                            if fy_end_month is None:
                                fy_end_month = max(q_months)
                    
                    if fy_end_month is None:
                        fy_end_month = 12
                    
                    q_month = best_quarter_end.month
                    q_year = best_quarter_end.year
                    
                    months_after_fy = (q_month - fy_end_month) % 12
                    
                    if months_after_fy == 0:
                        q_num = 4
                    elif months_after_fy <= 3:
                        q_num = 1
                    elif months_after_fy <= 6:
                        q_num = 2
                    else:
                        q_num = 3
                    
                    if q_month > fy_end_month:
                        fy = q_year + 1
                    else:
                        fy = q_year
                    
                    return f"Q{q_num} FY{str(fy)[2:]}"
        except Exception as e:
            print(f"    earnings_history error: {e}")
        
        return None
        
    except Exception as e:
        print(f"    Error getting fiscal quarter for {ticker}: {e}")
        return None


# ------------------------------------
# MAIN EARNINGS DATA FUNCTION
# ------------------------------------

def get_earnings_data(ticker, earnings_date=None):
    """Get all earnings-related data for a ticker."""
    result = {
        "Fiscal Quarter": None, "EPS Estimate": None, "Reported EPS": None, "EPS Surprise (%)": None, 
        "Earnings Date": None, "Earnings Date (yfinance)": None, "Date Check": "",
        "Earnings Released": None, "Earnings Timing": None
    }
    
    try:
        info = yf.Ticker(ticker).info
        
        if earnings_date is None:
            ts = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
            if ts: earnings_date = datetime.fromtimestamp(ts)
        
        if earnings_date:
            result["Earnings Date"] = earnings_date
            result["Earnings Released"] = earnings_date.date() < datetime.today().date()
            result["Fiscal Quarter"] = get_fiscal_quarter(ticker, earnings_date)
        
        # Get yfinance earnings date
        print(f"    Checking yfinance...")
        yf_date, yf_timing = get_yfinance_earnings_date(ticker)
        if yf_date:
            result["Earnings Date (yfinance)"] = yf_date
            date_check = check_date_status(earnings_date, yf_date)
            result["Date Check"] = date_check
            print(f"      yfinance: {yf_date.strftime('%Y-%m-%d')} | Date Check: {date_check}")
            if date_check == "DATE PASSED":
                print(f"      ⚠️ Finviz={earnings_date.date() if earnings_date else 'N/A'}, yfinance={yf_date.date()}")
        else:
            print(f"      yfinance: not found")
            result["Date Check"] = ""
        
        # Get EPS if released
        if result["Earnings Released"]:
            print(f"    Getting EPS for {result['Fiscal Quarter']}...")
            eps = get_eps_from_yfinance(ticker, earnings_date)
            if eps["EPS Estimate"] is not None or eps["Reported EPS"] is not None:
                result.update({k: eps[k] for k in ["EPS Estimate", "Reported EPS", "EPS Surprise (%)"]})
                if eps["Earnings Timing"]: result["Earnings Timing"] = eps["Earnings Timing"]
                print(f"      Est={result['EPS Estimate']}, Act={result['Reported EPS']}")
            else:
                cal = scrape_yahoo_calendar(earnings_date)
                if ticker.upper() in cal:
                    d = cal[ticker.upper()]
                    result.update({k: d.get(k) for k in ["EPS Estimate", "Reported EPS", "EPS Surprise (%)"]})
                    if d.get("Earnings Timing"): result["Earnings Timing"] = d["Earnings Timing"]
        
        if not result["Earnings Timing"]:
            if yf_timing:
                result["Earnings Timing"] = yf_timing
            elif result["Earnings Date"]:
                h = result["Earnings Date"].hour
                result["Earnings Timing"] = "BMO" if h < 10 else ("AMC" if h >= 16 else None)
    except Exception as e:
        print(f"    Error: {str(e)}")
    
    return result


# ------------------------------------
# HELPER FUNCTIONS
# ------------------------------------

def safe_dt(v):
    if pd.isna(v) or v is None: return pd.NaT
    if isinstance(v, datetime): return v
    if isinstance(v, pd.Timestamp): return v.to_pydatetime()
    try: return pd.to_datetime(v).to_pydatetime()
    except: return pd.NaT

def to_num(x):
    try: return float(str(x).replace(",", "").replace("$", "").strip())
    except: return np.nan

def parse_mcap(v):
    if pd.isna(v): return np.nan
    v = str(v).upper().replace("$", "").strip()
    try:
        if v.endswith("B"): return float(v[:-1]) * 1e9
        if v.endswith("M"): return float(v[:-1]) * 1e6
        if v.endswith("K"): return float(v[:-1]) * 1e3
        return float(v)
    except: return np.nan

def get_base(px, d, timing):
    try:
        d = safe_dt(d)
        if pd.isna(d): return np.nan
        if timing == 'BMO':
            p = px.loc[px.index <= d - timedelta(days=1)]
            return p.iloc[-1] if not p.empty else np.nan
        p = px.loc[px.index <= d]
        return p.iloc[-1] if not p.empty else np.nan
    except: return np.nan

def get_ret(px, d, days, timing):
    try:
        d = safe_dt(d)
        base = get_base(px, d, timing)
        if pd.isna(base): return np.nan
        after = px.loc[px.index > d]
        if len(after) < days: return np.nan
        return (after.iloc[days-1] / base) - 1
    except: return np.nan

def get_ret_today(px, d, timing):
    try:
        d = safe_dt(d)
        base = get_base(px, d, timing)
        if pd.isna(base): return np.nan
        after = px.loc[px.index > d]
        return (after.iloc[-1] / base) - 1 if not after.empty else np.nan
    except: return np.nan

def get_stats(px, d, days, timing):
    try:
        d = safe_dt(d)
        base = get_base(px, d, timing)
        if pd.isna(base): return (np.nan,)*5
        after = px.loc[px.index > d].head(days)
        if after.empty: return (np.nan,)*5
        hi, lo = after.max(), after.min()
        return ((hi/base)-1, (lo/base)-1, (hi-lo)/base, hi, lo)
    except: return (np.nan,)*5


# ------------------------------------
# BACKFILL EXISTING DATA
# ------------------------------------

def backfill(df):
    """Backfill missing data in existing returns tracker."""
    if "Earnings Date (yfinance)" not in df.columns:
        df["Earnings Date (yfinance)"] = pd.NaT
    df["Earnings Date (yfinance)"] = pd.to_datetime(df["Earnings Date (yfinance)"], errors='coerce')
    
    for col in ["EPS Estimate", "Reported EPS", "EPS Surprise (%)", "Fiscal Quarter", "Earnings Timing", "Date Added", "Date Check"]:
        if col not in df.columns: 
            df[col] = np.nan if col in ["EPS Estimate", "Reported EPS", "EPS Surprise (%)"] else ""
    
    def needs_bf(row):
        fq = row.get("Fiscal Quarter")
        fq_needs_update = pd.isna(fq) or fq == "" or not (isinstance(fq, str) and "Q" in fq and "FY" in fq)
        
        eps_needs_update = False
        if pd.isna(row.get("EPS Estimate")) and pd.isna(row.get("Reported EPS")):
            ed = safe_dt(row.get("Earnings Date"))
            if pd.notna(ed) and ed.date() < datetime.today().date():
                eps_needs_update = True
        
        yf_date_needs_update = pd.isna(row.get("Earnings Date (yfinance)"))
        
        date_check_needs_update = (pd.isna(row.get("Date Check")) or row.get("Date Check") == "") and pd.notna(row.get("Earnings Date (yfinance)"))
        
        return fq_needs_update or eps_needs_update or yf_date_needs_update or date_check_needs_update
    
    need = df[df.apply(needs_bf, axis=1)]
    if need.empty:
        print("  No backfill needed.")
        return df
    
    print(f"\n🔄 Backfilling {len(need)} entries...")
    
    for idx, row in need.iterrows():
        t = row["Ticker"]
        ed = safe_dt(row.get("Earnings Date"))
        fq = row.get("Fiscal Quarter")
        
        try:
            if pd.notna(ed):
                if pd.isna(fq) or fq == "" or not (isinstance(fq, str) and "Q" in fq and "FY" in fq):
                    new_fq = get_fiscal_quarter(t, ed)
                    if new_fq:
                        df.at[idx, "Fiscal Quarter"] = new_fq
                        print(f"  {t} → {new_fq}")
                
                if pd.isna(row.get("Earnings Date (yfinance)")):
                    yf_date, _ = get_yfinance_earnings_date(t)
                    if yf_date:
                        df.at[idx, "Earnings Date (yfinance)"] = pd.Timestamp(yf_date)
                        date_check = check_date_status(ed, yf_date)
                        df.at[idx, "Date Check"] = date_check
                        print(f"    {t}: yfinance date added, Date Check: {date_check}")
                else:
                    yf_existing = df.at[idx, "Earnings Date (yfinance)"]
                    if pd.notna(yf_existing) and (pd.isna(row.get("Date Check")) or row.get("Date Check") == ""):
                        date_check = check_date_status(ed, yf_existing)
                        df.at[idx, "Date Check"] = date_check
                
                if ed.date() < datetime.today().date():
                    if pd.isna(row.get("EPS Estimate")) and pd.isna(row.get("Reported EPS")):
                        eps = get_eps_from_yfinance(t, ed)
                        if eps["EPS Estimate"] is not None:
                            df.at[idx, "EPS Estimate"] = eps["EPS Estimate"]
                        if eps["Reported EPS"] is not None:
                            df.at[idx, "Reported EPS"] = eps["Reported EPS"]
                        if eps["EPS Surprise (%)"] is not None:
                            df.at[idx, "EPS Surprise (%)"] = eps["EPS Surprise (%)"]
                        if eps["Earnings Timing"]:
                            df.at[idx, "Earnings Timing"] = eps["Earnings Timing"]
                        
                        if pd.isna(df.at[idx, "EPS Estimate"]) and pd.isna(df.at[idx, "Reported EPS"]):
                            cal = scrape_yahoo_calendar(ed)
                            if t.upper() in cal:
                                d = cal[t.upper()]
                                for k in ["EPS Estimate", "Reported EPS", "EPS Surprise (%)", "Earnings Timing"]:
                                    if d.get(k) is not None:
                                        df.at[idx, k] = d[k]
                
                time.sleep(0.3)
        except Exception as e:
            print(f"  {t} error: {e}")
    
    print(f"  ✅ Done")
    return df


# ------------------------------------
# BUILD UNIVERSE (finds new tickers)
# ------------------------------------

def build_universe():
    """Scan Finviz for new tickers meeting criteria."""
    today = datetime.today().date()
    today_str = today.strftime("%Y-%m-%d")
    
    tickers = get_all_tickers()
    print(f"[{today}] Found {len(tickers)} Finviz tickers...")

    # Filter by Barchart buy signal
    filtered = []
    for t in tickers:
        if has_buy_signal(t): filtered.append(t)
        time.sleep(0.3)
    print(f"[{today}] {len(filtered)} passed Barchart filter.")

    # Get existing tickers from returns tracker
    existing_tickers = set()
    if os.path.exists(RETURNS_FILE):
        existing_df = pd.read_csv(RETURNS_FILE)
        existing_tickers = set(existing_df["Ticker"].dropna().unique())

    rows, skipped = [], []
    for t in filtered:
        row = get_finviz_stats(t)
        print(f"  {t}: Getting earnings...")
        e = get_earnings_data(t)
        
        if e.get("Earnings Released"):
            print(f"    ⏭️ SKIPPED: Earnings Released")
            skipped.append((t, "RELEASED"))
            time.sleep(0.3)
            continue
        
        row.update({k: e.get(k) for k in [
            "Earnings Date", "EPS Estimate", "Reported EPS", "EPS Surprise (%)", 
            "Earnings Timing", "Fiscal Quarter", "Earnings Date (yfinance)", "Date Check"
        ]})
        
        if t not in existing_tickers:
            row["Date Added"] = today_str
            print(f"    ✅ NEW: {row.get('Company Name')} | {e.get('Fiscal Quarter')} | Added: {today_str}")
        else:
            row["Date Added"] = None
            print(f"    ✅ {row.get('Company Name')} | {e.get('Fiscal Quarter')}")
        
        rows.append(row)
        time.sleep(0.5)
    
    if skipped:
        print(f"\n📋 Skipped {len(skipped)} tickers:")
        for t, reason in skipped:
            print(f"   - {t}: {reason}")

    return rows, skipped


# ------------------------------------
# UPDATE RETURNS
# ------------------------------------

def update_returns(new_rows):
    """Update returns tracker with new data and calculate returns."""
    
    # Load existing data
    if os.path.exists(RETURNS_FILE):
        existing_df = pd.read_csv(RETURNS_FILE)
        existing_df["Earnings Date"] = pd.to_datetime(existing_df["Earnings Date"], errors='coerce')
        existing_df["Earnings Date (yfinance)"] = pd.to_datetime(existing_df.get("Earnings Date (yfinance)"), errors='coerce')
    else:
        existing_df = pd.DataFrame()
    
    # Add new rows
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df["Earnings Date"] = pd.to_datetime(new_df["Earnings Date"], errors='coerce')
        new_df["Earnings Date (yfinance)"] = pd.to_datetime(new_df["Earnings Date (yfinance)"], errors='coerce')
        
        for c in ["Price", "EPS (TTM)", "P/E", "Forward P/E", "Beta", "EPS Estimate", "Reported EPS", "EPS Surprise (%)"]:
            if c in new_df.columns: new_df[c] = new_df[c].apply(to_num)
        if "Market Cap" in new_df.columns:
            new_df["Market Cap"] = new_df["Market Cap"].apply(parse_mcap)
        
        # Preserve Date Added from existing
        if not existing_df.empty:
            for idx, row in new_df.iterrows():
                ticker, fq = row["Ticker"], row.get("Fiscal Quarter")
                match = existing_df[(existing_df["Ticker"] == ticker) & (existing_df["Fiscal Quarter"] == fq)]
                if not match.empty and pd.notna(match.iloc[0].get("Date Added")) and match.iloc[0].get("Date Added") != "":
                    new_df.at[idx, "Date Added"] = match.iloc[0]["Date Added"]
        
        # Combine
        if not existing_df.empty:
            udf = pd.concat([existing_df, new_df], ignore_index=True)
            udf.drop_duplicates(subset=["Ticker", "Fiscal Quarter"], keep="last", inplace=True)
        else:
            udf = new_df
    else:
        udf = existing_df
    
    if udf.empty:
        print("\n⚠️ No data to process.")
        return pd.DataFrame()
    
    # Backfill missing data
    udf = backfill(udf)
    
    # Calculate returns
    tickers = udf["Ticker"].dropna().unique().tolist()
    if not tickers:
        return udf
    
    valid = udf["Earnings Date"].dropna()
    if valid.empty:
        return udf
    
    start = pd.to_datetime(valid.min()) - timedelta(days=5)
    end = datetime.today() + timedelta(days=2)
    
    print(f"\nDownloading prices for {len(tickers)} tickers...")
    px = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)
    
    rows = []
    for _, r in udf.iterrows():
        t, d = r["Ticker"], safe_dt(r["Earnings Date"])
        timing = r.get("Earnings Timing")
        if pd.isna(d): continue
        
        try:
            p = px[t]["Close"].dropna() if len(tickers) > 1 else px["Close"].dropna()
            base = get_base(p, d, timing)
            hi, lo, rng, hip, lop = get_stats(p, d, 7, timing)
            
            rows.append({
                "Ticker": t, "Fiscal Quarter": r.get("Fiscal Quarter"), 
                "Earnings Date": d, "Earnings Date (yfinance)": r.get("Earnings Date (yfinance)"),
                "Date Check": r.get("Date Check"), "Date Added": r.get("Date Added"),
                "Earnings Timing": timing,
                "Price": round(base, 2) if pd.notna(base) else np.nan,
                "EPS (TTM)": r.get("EPS (TTM)"), "EPS Estimate": r.get("EPS Estimate"), 
                "Reported EPS": r.get("Reported EPS"), "EPS Surprise (%)": r.get("EPS Surprise (%)"),
                "P/E": r.get("P/E"), "Forward P/E": r.get("Forward P/E"), 
                "Market Cap": r.get("Market Cap"), "Beta": r.get("Beta"),
                "1D Return": round(get_ret(p, d, 1, timing), 3) if pd.notna(get_ret(p, d, 1, timing)) else np.nan,
                "3D Return": round(get_ret(p, d, 3, timing), 3) if pd.notna(get_ret(p, d, 3, timing)) else np.nan,
                "5D Return": round(get_ret(p, d, 5, timing), 3) if pd.notna(get_ret(p, d, 5, timing)) else np.nan,
                "7D Return": round(get_ret(p, d, 7, timing), 3) if pd.notna(get_ret(p, d, 7, timing)) else np.nan,
                "10D Return": round(get_ret(p, d, 10, timing), 3) if pd.notna(get_ret(p, d, 10, timing)) else np.nan,
                "Return to Today": round(get_ret_today(p, d, timing), 3) if pd.notna(get_ret_today(p, d, timing)) else np.nan,
                "1W High Return": round(hi, 3) if pd.notna(hi) else np.nan, 
                "1W Low Return": round(lo, 3) if pd.notna(lo) else np.nan,
                "1W Range": round(rng, 3) if pd.notna(rng) else np.nan, 
                "1W High Price": round(hip, 2) if pd.notna(hip) else np.nan, 
                "1W Low Price": round(lop, 2) if pd.notna(lop) else np.nan,
                "Company Name": r.get("Company Name", t), "Sector": r.get("Sector", "N/A")
            })
        except: continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors='coerce')
        df["Earnings Date (yfinance)"] = pd.to_datetime(df["Earnings Date (yfinance)"], errors='coerce')
        df = df.sort_values("Earnings Date").reset_index(drop=True)
    
    df.to_csv(RETURNS_FILE, index=False)
    print(f"✅ Saved {RETURNS_FILE} ({len(df)} entries)")
    
    return df


# ------------------------------------
# EXPORT HOURLY PRICES FOR POWERBI
# ------------------------------------

def export_hourly_prices(udf, days_after=5):
    """Export hourly prices for each ticker for PowerBI charting."""
    if udf is None or udf.empty:
        print("No data to export prices.")
        return
    
    udf = udf.copy()
    udf["Earnings Date"] = pd.to_datetime(udf["Earnings Date"], errors='coerce')
    
    valid = udf[udf["Earnings Date"].notna()].copy()
    if valid.empty:
        print("No valid earnings dates for price export.")
        return
    
    tickers = valid["Ticker"].unique().tolist()
    
    print(f"\n📈 Exporting HOURLY prices for {len(tickers)} tickers...")
    print(f"   Range: {days_after} trading days after earnings")
    
    rows = []
    
    for _, r in valid.iterrows():
        t = r["Ticker"]
        earnings_date = r["Earnings Date"]
        fiscal_quarter = r.get("Fiscal Quarter", "")
        company_name = r.get("Company Name", t)
        timing = r.get("Earnings Timing", "")
        
        try:
            start_date = earnings_date - timedelta(days=7)
            end_date = earnings_date + timedelta(days=days_after + 5)
            
            today = datetime.today()
            if end_date > today:
                end_date = today + timedelta(days=1)
            
            print(f"   {t}: Downloading hourly data...")
            
            ticker_obj = yf.Ticker(t)
            hourly_data = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1h"
            )
            
            if hourly_data.empty:
                print(f"      No hourly data available")
                continue
            
            prices = hourly_data["Close"].dropna()
            
            if prices.empty:
                print(f"      No close prices")
                continue
            
            prices.index = prices.index.tz_localize(None)
            
            earnings_dt = pd.Timestamp(earnings_date).tz_localize(None)
            earnings_day = earnings_dt.date()
            
            if timing == "BMO":
                base_prices = prices[prices.index.date < earnings_day]
                if base_prices.empty:
                    print(f"      No base price found for BMO")
                    continue
                base_price = base_prices.iloc[-1]
                base_time = base_prices.index[-1]
                track_start = prices[prices.index.date >= earnings_day]
                if track_start.empty:
                    print(f"      No tracking data for BMO")
                    continue
                track_start_time = track_start.index[0]
            else:
                day_prices = prices[prices.index.date == earnings_day]
                if day_prices.empty:
                    print(f"      No base price found for AMC")
                    continue
                base_price = day_prices.iloc[-1]
                base_time = day_prices.index[-1]
                track_start = prices[prices.index.date > earnings_day]
                if track_start.empty:
                    print(f"      No tracking data for AMC")
                    continue
                track_start_time = track_start.index[0]
            
            print(f"      Base: ${base_price:.2f} at {base_time}")
            print(f"      Tracking from: {track_start_time}")
            
            trading_days_count = 0
            last_trading_date = None
            
            track_start_cmp = track_start_time.to_pydatetime() if hasattr(track_start_time, 'to_pydatetime') else track_start_time
            if hasattr(track_start_cmp, 'tzinfo') and track_start_cmp.tzinfo is not None:
                track_start_cmp = track_start_cmp.replace(tzinfo=None)
            
            for dt_idx, close_price in prices.items():
                if hasattr(dt_idx, 'to_pydatetime'):
                    dt = dt_idx.to_pydatetime()
                else:
                    dt = pd.to_datetime(dt_idx).to_pydatetime()
                
                if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                
                if dt < track_start_cmp:
                    continue
                
                current_date = dt.date()
                if last_trading_date is None:
                    trading_days_count = 1
                    last_trading_date = current_date
                elif current_date > last_trading_date:
                    trading_days_count += 1
                    last_trading_date = current_date
                
                if trading_days_count > days_after:
                    break
                
                base_time_cmp = base_time.to_pydatetime() if hasattr(base_time, 'to_pydatetime') else base_time
                if hasattr(base_time_cmp, 'tzinfo') and base_time_cmp.tzinfo is not None:
                    base_time_cmp = base_time_cmp.replace(tzinfo=None)
                hours_from_earnings = (dt - base_time_cmp).total_seconds() / 3600
                
                return_from_earnings = ((close_price / base_price) - 1) * 100
                
                rows.append({
                    "Ticker": t,
                    "Company Name": company_name,
                    "Fiscal Quarter": fiscal_quarter,
                    "Datetime": dt,
                    "Date": dt.date(),
                    "Time": dt.strftime("%H:%M"),
                    "Hour": dt.hour,
                    "Close": round(close_price, 2),
                    "Earnings Date": earnings_date.date() if hasattr(earnings_date, 'date') else earnings_date,
                    "Earnings Timing": timing,
                    "Base Price": round(base_price, 2),
                    "Hours From Earnings": round(hours_from_earnings, 1),
                    "Trading Day": trading_days_count,
                    "Return From Earnings (%)": round(return_from_earnings, 2)
                })
            
            print(f"      Collected {len([r for r in rows if r['Ticker'] == t])} hourly points")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   Error processing {t}: {e}")
            continue
    
    if not rows:
        print("   No hourly price data collected.")
        return
    
    df = pd.DataFrame(rows)
    df = df.sort_values(["Ticker", "Fiscal Quarter", "Datetime"]).reset_index(drop=True)
    
    df.to_csv(PRICES_FILE, index=False)
    print(f"✅ Saved {PRICES_FILE} ({len(df)} rows, {df['Ticker'].nunique()} tickers)")
    
    print(f"\n📊 Summary:")
    print(f"   Total hourly data points: {len(df)}")
    print(f"   Unique tickers: {df['Ticker'].nunique()}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Avg points per ticker: {len(df) / df['Ticker'].nunique():.0f}")


# ------------------------------------
# MAIN
# ------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("EARNINGS MOMENTUM v17")
    print(f"📅 {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Step 1: Find new tickers from Finviz + Barchart
    new_rows, skipped = build_universe()
    
    # Step 2: Update returns tracker with new data + calculate returns
    returns_df = update_returns(new_rows)
    
    # Step 3: Export hourly prices for PowerBI
    export_hourly_prices(returns_df, days_after=5)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)