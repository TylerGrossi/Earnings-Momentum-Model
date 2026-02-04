import yfinance as yf
import pandas as pd
import os

# Create a directory for the output
folder_name = "AMD_Data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Initialize Ticker
amd = yf.Ticker("AMD")

print("Fetching data and saving to CSV...")

# 1. Historical Price Data (The biggest table)
amd.history(period="max").to_csv(f"{folder_name}/amd_history.csv")

# 2. Financial Statements
amd.income_stmt.to_csv(f"{folder_name}/amd_income_statement.csv")
amd.balance_sheet.to_csv(f"{folder_name}/amd_balance_sheet.csv")
amd.cashflow.to_csv(f"{folder_name}/amd_cashflow.csv")

# 3. Key Statistics and Company Info
# Since .info is a dictionary, we convert it to a DataFrame first
info_df = pd.DataFrame.from_dict(amd.info, orient='index', columns=['Value'])
info_df.to_csv(f"{folder_name}/amd_info.csv")

# 4. Holders and Analyst Data
amd.major_holders.to_csv(f"{folder_name}/amd_major_holders.csv")
amd.institutional_holders.to_csv(f"{folder_name}/amd_institutional_holders.csv")
amd.recommendations.to_csv(f"{folder_name}/amd_analyst_recommendations.csv")

# 5. Calendar (Earnings dates)
if amd.calendar is not None:
    amd.calendar.to_csv(f"{folder_name}/amd_calendar.csv")

print(f"Done! All files are saved in the '{folder_name}' folder.")