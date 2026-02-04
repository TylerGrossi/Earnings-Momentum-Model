import yfinance as yf
import pandas as pd
import os

folder_name = "AMD_Data"
os.makedirs(folder_name, exist_ok=True)

amd = yf.Ticker("AMD")

# List of attributes that are already DataFrames
df_attributes = {
    "history": amd.history(period="max"),
    "income_stmt": amd.income_stmt,
    "balance_sheet": amd.balance_sheet,
    "cashflow": amd.cashflow,
    "major_holders": amd.major_holders,
    "institutional_holders": amd.institutional_holders,
    "recommendations": amd.recommendations
}

# Save DataFrames
for name, df in df_attributes.items():
    if df is not None and not df.empty:
        df.to_csv(f"{folder_name}/amd_{name}.csv")

# List of attributes that are Dictionaries (Need conversion)
dict_attributes = {
    "info": amd.info,
    "calendar": amd.calendar
}

# Save Dictionaries
for name, data in dict_attributes.items():
    if data:
        temp_df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
        temp_df.to_csv(f"{folder_name}/amd_{name}.csv")

print(f"Success! All AMD data is now in the '{folder_name}' folder.")