"""
MCP Server for Earnings Momentum Strategy
==========================================
Exposes strategy data and analysis tools via Model Context Protocol.

Run standalone: python mcp_server.py
Or start from Streamlit tab.
"""

import json
import sys
from datetime import datetime, timedelta
from typing import Any
import pandas as pd
import numpy as np

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ------------------------------------
# DATA LOADING (standalone mode)
# ------------------------------------

def load_returns_data():
    """Load returns tracker data from GitHub or local."""
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/returns_tracker.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/returns_tracker.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and '1D Return' in df.columns:
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    try:
        df = pd.read_csv('returns_tracker.csv')
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
        return df
    except:
        return pd.DataFrame()


def load_hourly_prices():
    """Load hourly prices data from GitHub or local."""
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/hourly_prices.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/hourly_prices.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and 'Trading Day' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    try:
        df = pd.read_csv('hourly_prices.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
        return df
    except:
        return pd.DataFrame()


def filter_data(returns_df):
    """Apply standard filtering: remove DATE PASSED and require 5D Return."""
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    
    df = returns_df.copy()
    
    # Remove DATE PASSED
    if 'Date Check' in df.columns:
        df = df[df['Date Check'] != 'DATE PASSED']
    
    # Require valid 5D Return
    if '5D Return' in df.columns:
        df = df[df['5D Return'].notna()]
    
    return df


# ------------------------------------
# TOOL IMPLEMENTATIONS
# ------------------------------------

def get_strategy_rules() -> dict:
    """Return the strategy rules and parameters."""
    return {
        "name": "Earnings Momentum Strategy",
        "entry_criteria": {
            "earnings_timing": "Stock has earnings scheduled for current week",
            "sma_crossover": "SMA20 > SMA50 (golden cross)",
            "barchart_signal": "Buy opinion on Barchart.com"
        },
        "entry_timing": {
            "BMO": "Buy at previous day's close (~4pm day before)",
            "AMC": "Buy at earnings day close (~4pm on earnings day)"
        },
        "exit_rules": {
            "stop_loss": "-10% from entry (take actual gap if opens lower)",
            "time_exit": "Day 5 close if stop not triggered",
            "profit_cap": "None - let winners run"
        },
        "position_sizing": "Equal weight per trade"
    }


def get_earnings_this_week() -> dict:
    """Get stocks with earnings this week from the tracker."""
    df = load_returns_data()
    if df.empty:
        return {"error": "Could not load data", "stocks": []}
    
    today = datetime.today()
    days_since_sunday = (today.weekday() + 1) % 7
    week_start = today - timedelta(days=days_since_sunday)
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    this_week = df[
        (df['Earnings Date'] >= week_start) & 
        (df['Earnings Date'] <= week_end)
    ].copy()
    
    stocks = []
    for _, row in this_week.iterrows():
        stocks.append({
            "ticker": row.get('Ticker', 'N/A'),
            "company": row.get('Company Name', 'N/A'),
            "earnings_date": str(row.get('Earnings Date', 'N/A'))[:10],
            "timing": row.get('Earnings Timing', 'N/A'),
            "price": row.get('Price', 'N/A'),
            "date_check": row.get('Date Check', 'N/A'),
            "eps_estimate": row.get('EPS Estimate', 'N/A'),
            "reported_eps": row.get('Reported EPS', 'N/A'),
            "eps_surprise_pct": row.get('EPS Surprise (%)', 'N/A'),
            "5d_return": row.get('5D Return', 'N/A')
        })
    
    return {
        "week_range": f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}",
        "count": len(stocks),
        "stocks": stocks
    }


def get_stock_details(ticker: str) -> dict:
    """Get detailed information for a specific stock."""
    df = load_returns_data()
    if df.empty:
        return {"error": "Could not load data"}
    
    stock_data = df[df['Ticker'].str.upper() == ticker.upper()]
    if stock_data.empty:
        return {"error": f"Ticker {ticker} not found in database"}
    
    # Get most recent entry
    row = stock_data.sort_values('Earnings Date', ascending=False).iloc[0]
    
    result = {
        "ticker": row.get('Ticker'),
        "company_name": row.get('Company Name'),
        "sector": row.get('Sector'),
        "market_cap": row.get('Market Cap'),
        "earnings": {
            "date": str(row.get('Earnings Date', ''))[:10],
            "timing": row.get('Earnings Timing'),
            "fiscal_quarter": row.get('Fiscal Quarter'),
            "eps_estimate": row.get('EPS Estimate'),
            "reported_eps": row.get('Reported EPS'),
            "eps_surprise_pct": row.get('EPS Surprise (%)'),
        },
        "entry": {
            "price": row.get('Price'),
            "date_added": row.get('Date Added'),
            "date_check": row.get('Date Check')
        },
        "returns": {
            "1d": row.get('1D Return'),
            "2d": row.get('2D Return'),
            "3d": row.get('3D Return'),
            "4d": row.get('4D Return'),
            "5d": row.get('5D Return'),
        },
        "technicals": {
            "beta": row.get('Beta'),
            "pe_ratio": row.get('P/E'),
        }
    }
    
    # Convert numpy types to Python types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if not pd.isna(obj) else None
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if pd.isna(obj):
            return None
        return obj
    
    return convert(result)


def get_strategy_performance() -> dict:
    """Get overall strategy performance metrics."""
    df = filter_data(load_returns_data())
    if df.empty:
        return {"error": "Could not load data"}
    
    total_trades = len(df)
    
    # 5D Return stats
    returns_5d = df['5D Return'].dropna() * 100  # Convert to percentage
    
    # Win rate
    wins = (returns_5d > 0).sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    # EPS surprise stats
    eps_surprise = df['EPS Surprise (%)'].dropna() if 'EPS Surprise (%)' in df.columns else pd.Series()
    beat_rate = (eps_surprise > 0).mean() * 100 if len(eps_surprise) > 0 else None
    
    # Sector breakdown
    sector_stats = {}
    if 'Sector' in df.columns:
        for sector in df['Sector'].dropna().unique():
            sector_df = df[df['Sector'] == sector]
            sector_returns = sector_df['5D Return'].dropna() * 100
            if len(sector_returns) > 0:
                sector_stats[sector] = {
                    "count": len(sector_df),
                    "avg_return": round(sector_returns.mean(), 2),
                    "win_rate": round((sector_returns > 0).mean() * 100, 1)
                }
    
    return {
        "total_trades": total_trades,
        "returns_5d": {
            "total": round(returns_5d.sum(), 2),
            "average": round(returns_5d.mean(), 2),
            "median": round(returns_5d.median(), 2),
            "std_dev": round(returns_5d.std(), 2),
            "min": round(returns_5d.min(), 2),
            "max": round(returns_5d.max(), 2),
        },
        "win_rate_pct": round(win_rate, 1),
        "beat_rate_pct": round(beat_rate, 1) if beat_rate else None,
        "sector_breakdown": sector_stats
    }


def run_backtest(stop_loss_pct: float = -10, holding_days: int = 5) -> dict:
    """
    Run a backtest with custom parameters.
    
    Args:
        stop_loss_pct: Stop loss percentage (e.g., -10 for -10%)
        holding_days: Maximum days to hold (1-10)
    """
    returns_df = filter_data(load_returns_data())
    hourly_df = load_hourly_prices()
    
    if returns_df.empty or hourly_df.empty:
        return {"error": "Could not load data for backtest"}
    
    # Validate inputs
    stop_loss_pct = max(-50, min(-1, stop_loss_pct))  # Clamp between -50 and -1
    holding_days = max(1, min(10, holding_days))
    
    stop_loss = stop_loss_pct / 100.0
    
    hourly_df = hourly_df.copy()
    hourly_df['Earnings Date'] = pd.to_datetime(hourly_df['Earnings Date']).dt.date
    returns_df = returns_df.copy()
    returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date']).dt.date
    
    today = datetime.now().date()
    
    # Only completed trades
    valid_trades = returns_df[
        (returns_df['5D Return'].notna()) & 
        (returns_df['Earnings Date'] <= (today - timedelta(days=7)))
    ]
    
    results = []
    for _, trade in valid_trades.iterrows():
        ticker = trade['Ticker']
        e_date = trade['Earnings Date']
        normal_return = trade['5D Return']
        
        trade_data = hourly_df[
            (hourly_df['Ticker'] == ticker) & 
            (hourly_df['Earnings Date'] == e_date) &
            (hourly_df['Trading Day'] >= 1)
        ].sort_values('Datetime')
        
        if trade_data.empty:
            continue
        
        exit_day = min(holding_days, trade_data['Trading Day'].max())
        exit_day_data = trade_data[trade_data['Trading Day'] == exit_day]
        if exit_day_data.empty:
            continue
        
        close_return = exit_day_data['Return From Earnings (%)'].iloc[-1] / 100
        
        # Simulate stop loss
        final_return = close_return
        exit_reason = "held_to_exit"
        first_candle = True
        
        for _, hour in trade_data.iterrows():
            if int(hour['Trading Day']) > exit_day:
                break
            
            h_ret = hour['Return From Earnings (%)'] / 100
            
            if h_ret <= stop_loss:
                # Gap down handling
                if first_candle and h_ret < stop_loss:
                    final_return = h_ret
                    exit_reason = "gap_down"
                else:
                    final_return = stop_loss
                    exit_reason = "stop_loss"
                break
            
            first_candle = False
        
        results.append({
            "ticker": ticker,
            "earnings_date": str(e_date),
            "normal_return": normal_return,
            "strategy_return": final_return,
            "exit_reason": exit_reason
        })
    
    if not results:
        return {"error": "No valid trades for backtest"}
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    normal_total = results_df['normal_return'].sum() * 100
    strategy_total = results_df['strategy_return'].sum() * 100
    alpha = strategy_total - normal_total
    
    exit_breakdown = results_df['exit_reason'].value_counts().to_dict()
    
    wins = (results_df['strategy_return'] > 0).sum()
    win_rate = wins / len(results_df) * 100
    
    return {
        "parameters": {
            "stop_loss_pct": stop_loss_pct,
            "holding_days": holding_days
        },
        "total_trades": len(results_df),
        "normal_model": {
            "total_return_pct": round(normal_total, 2),
            "avg_return_pct": round(results_df['normal_return'].mean() * 100, 2)
        },
        "strategy_model": {
            "total_return_pct": round(strategy_total, 2),
            "avg_return_pct": round(results_df['strategy_return'].mean() * 100, 2),
            "alpha_pct": round(alpha, 2)
        },
        "win_rate_pct": round(win_rate, 1),
        "exit_breakdown": exit_breakdown,
        "trades": results[:10]  # Return first 10 trades as sample
    }


def get_beat_miss_analysis() -> dict:
    """Analyze performance by earnings beat vs miss."""
    df = filter_data(load_returns_data())
    if df.empty:
        return {"error": "Could not load data"}
    
    if 'EPS Surprise (%)' not in df.columns:
        return {"error": "No EPS surprise data available"}
    
    df = df[df['EPS Surprise (%)'].notna()].copy()
    df['5D Return Pct'] = df['5D Return'] * 100
    
    beats = df[df['EPS Surprise (%)'] > 0]
    misses = df[df['EPS Surprise (%)'] < 0]
    inline = df[df['EPS Surprise (%)'] == 0]
    
    def calc_stats(subset, label):
        if len(subset) == 0:
            return None
        returns = subset['5D Return Pct']
        return {
            "category": label,
            "count": len(subset),
            "total_return_pct": round(returns.sum(), 2),
            "avg_return_pct": round(returns.mean(), 2),
            "median_return_pct": round(returns.median(), 2),
            "win_rate_pct": round((returns > 0).mean() * 100, 1),
            "avg_eps_surprise_pct": round(subset['EPS Surprise (%)'].mean(), 2)
        }
    
    return {
        "total_with_eps_data": len(df),
        "beat": calc_stats(beats, "Beat"),
        "miss": calc_stats(misses, "Miss"),
        "inline": calc_stats(inline, "Inline"),
        "insight": "Compare average returns between beats and misses to see if the strategy benefits from positive earnings surprises."
    }


def compare_stop_losses() -> dict:
    """Compare different stop loss levels from -2% to -20%."""
    returns_df = filter_data(load_returns_data())
    hourly_df = load_hourly_prices()
    
    if returns_df.empty or hourly_df.empty:
        return {"error": "Could not load data"}
    
    results = []
    for sl_pct in range(-2, -22, -2):
        backtest = run_backtest(stop_loss_pct=sl_pct, holding_days=5)
        if "error" not in backtest:
            results.append({
                "stop_loss_pct": sl_pct,
                "total_return_pct": backtest["strategy_model"]["total_return_pct"],
                "alpha_pct": backtest["strategy_model"]["alpha_pct"],
                "win_rate_pct": backtest["win_rate_pct"]
            })
    
    if not results:
        return {"error": "Could not run backtests"}
    
    # Find best
    best = max(results, key=lambda x: x["total_return_pct"])
    
    return {
        "comparison": results,
        "best_stop_loss_pct": best["stop_loss_pct"],
        "best_total_return_pct": best["total_return_pct"],
        "recommendation": f"The {best['stop_loss_pct']}% stop loss provides the highest total return."
    }


# ------------------------------------
# MCP SERVER SETUP
# ------------------------------------

TOOLS = [
    {
        "name": "get_strategy_rules",
        "description": "Get the entry criteria, exit rules, and parameters of the Earnings Momentum Strategy",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_earnings_this_week", 
        "description": "Get all stocks with earnings scheduled for the current week (Sunday-Saturday) from the tracker",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_stock_details",
        "description": "Get detailed information about a specific stock including earnings data, returns, and technicals",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, MSFT)"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_strategy_performance",
        "description": "Get overall strategy performance metrics including total returns, win rate, and sector breakdown",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "run_backtest",
        "description": "Run a backtest with custom stop loss and holding period parameters",
        "input_schema": {
            "type": "object",
            "properties": {
                "stop_loss_pct": {
                    "type": "number",
                    "description": "Stop loss percentage, e.g., -10 for -10% stop loss. Range: -50 to -1"
                },
                "holding_days": {
                    "type": "integer", 
                    "description": "Maximum days to hold position. Range: 1-10"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_beat_miss_analysis",
        "description": "Analyze strategy performance by earnings beat vs miss categories",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "compare_stop_losses",
        "description": "Compare different stop loss levels from -2% to -20% to find optimal setting",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return JSON result."""
    try:
        if name == "get_strategy_rules":
            result = get_strategy_rules()
        elif name == "get_earnings_this_week":
            result = get_earnings_this_week()
        elif name == "get_stock_details":
            result = get_stock_details(arguments.get("ticker", ""))
        elif name == "get_strategy_performance":
            result = get_strategy_performance()
        elif name == "run_backtest":
            result = run_backtest(
                stop_loss_pct=arguments.get("stop_loss_pct", -10),
                holding_days=arguments.get("holding_days", 5)
            )
        elif name == "get_beat_miss_analysis":
            result = get_beat_miss_analysis()
        elif name == "compare_stop_losses":
            result = compare_stop_losses()
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def run_mcp_server():
    """Run the MCP server (async)."""
    if not MCP_AVAILABLE:
        print("MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
        return
    
    server = Server("earnings-momentum-strategy")
    
    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["input_schema"]
            )
            for t in TOOLS
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        result = execute_tool(name, arguments)
        return [TextContent(type="text", text=result)]
    
    @server.list_resources()
    async def list_resources():
        return [
            Resource(
                uri="earnings://strategy/rules",
                name="Strategy Rules",
                description="The entry and exit rules for the Earnings Momentum Strategy",
                mimeType="application/json"
            ),
            Resource(
                uri="earnings://data/returns",
                name="Returns Tracker",
                description="Historical returns data for all tracked stocks",
                mimeType="application/json"
            )
        ]
    
    @server.read_resource()
    async def read_resource(uri: str):
        if uri == "earnings://strategy/rules":
            return json.dumps(get_strategy_rules(), indent=2)
        elif uri == "earnings://data/returns":
            df = filter_data(load_returns_data())
            return df.to_json(orient="records", date_format="iso")
        return json.dumps({"error": "Unknown resource"})
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


# ------------------------------------
# MAIN
# ------------------------------------

if __name__ == "__main__":
    import asyncio
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode - run each tool and print results
        print("Testing MCP Tools...\n")
        
        for tool in TOOLS:
            print(f"=== {tool['name']} ===")
            if tool["name"] == "get_stock_details":
                result = execute_tool(tool["name"], {"ticker": "AAPL"})
            elif tool["name"] == "run_backtest":
                result = execute_tool(tool["name"], {"stop_loss_pct": -10, "holding_days": 5})
            else:
                result = execute_tool(tool["name"], {})
            
            # Truncate long results
            if len(result) > 500:
                print(result[:500] + "...\n[truncated]\n")
            else:
                print(result + "\n")
    else:
        # Run MCP server
        asyncio.run(run_mcp_server())