# Alpaca Algo Trading – Earnings Momentum Model

One script runs everything: reads your Earnings Momentum model’s `returns_tracker.csv`, finds today’s entry signals, and places **Market-on-Close** orders at 4pm ET. It can also run **automatically at 3:55 PM ET** every day so you don’t have to run it manually.

## Setup

1. **Install dependencies** (from this folder):

   ```bash
   cd "Alpaca Algo Trading"
   pip install -r requirements.txt
   ```

2. **Alpaca API keys**  
   Sign up at [Alpaca](https://alpaca.markets), open the [dashboard](https://app.alpaca.markets), and create **Paper** (or Live) API keys.

3. **Configure**  
   Add your Alpaca keys in **`.streamlit/secrets.toml`** (project root):
   ```toml
   ALPACA_API_KEY = "your_api_key_here"
   ALPACA_SECRET_KEY = "your_secret_key_here"
   PAPER = true
   ```

## One file: `alpaca_runner.py`

| Command | What it does |
|--------|-------------------------------|
| `python alpaca_runner.py` | Run once **now** (dry run, no orders). |
| `python alpaca_runner.py --live` | Run once **now** and submit MOC orders. |
| `python alpaca_runner.py --schedule` | Run at **3:55 PM ET every day** (dry run). Leave the window open. |
| `python alpaca_runner.py --schedule --live` | Run at **3:55 PM ET every day** and submit orders. Leave the window open. |

- **Run once**: Script runs immediately and exits. Use for testing or with Windows Task Scheduler.
- **Run on a schedule**: Script stays running, waits until 3:55 PM ET, runs the job, then waits until 3:55 PM ET the next day. Use on a PC or server that’s on all day (or a cloud VM). Press **Ctrl+C** to stop.

## Automatic 3:55 PM run (no daily manual run)

1. Open a terminal in the `Alpaca Algo Trading` folder.
2. Run:
   ```bash
   python alpaca_runner.py --schedule --live
   ```
3. Leave that window open (or run it in the background / as a service). Each day at 3:55 PM ET it will place MOC orders for that day’s signals and then wait until the next day.

**Alternative (Windows Task Scheduler)**  
If you prefer not to leave a process running, create a Windows task that runs daily at 3:55 PM and executes:
`python "C:\...\Alpaca Algo Trading\alpaca_runner.py" --live`  
(Use the full path to your Python and to the script.)

## Requirements

- Parent project’s `returns_tracker.csv` is updated (run your main Earnings Momentum model regularly so the tracker has current earnings and Date Check).
- For scheduled runs, the machine must be on and the script running (or the task scheduled) at 3:55 PM ET.

## Position sizing

Edit the top of `alpaca_runner.py`:

- `POSITION_FRACTION = 0.05` → 5% of equity per position.
- `FIXED_DOLLARS = 1000` → fixed $ per position; set to `None` to use the fraction.

## Troubleshooting

- **“No buy signals for today”** – Today isn’t an entry day for any ticker, or `returns_tracker.csv` is stale. Update the main model.
- **“Missing Alpaca credentials”** – Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.streamlit/secrets.toml`.
- **Scheduled run doesn’t fire** – Make sure the script is still running (e.g. no crash, PC not asleep at 3:55 PM ET).
