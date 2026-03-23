import pandas as pd
file_path = "backend/knowledge_base/raw/2024-v195-20032026-EU MRV Publication of information.xlsx"
dfraw = pd.read_excel(file_path, sheet_name="2024 Full ERs", skiprows=2)
for c in [
    'CO₂ emissions per transport work (mass) [g CO₂ / m tonnes · n miles]',
    'CO₂ emissions per transport work (freight) [g CO₂ / m tonnes · n miles]',
    'CO₂ emissions per transport work (dwt) [g CO₂ / dwt carried · n miles]'
]:
    count = dfraw[c].notna().sum()
    print(f"{c}: {count}")

print(f"Total rows: {len(dfraw)}")
dfraw['best_emission'] = dfraw['CO₂ emissions per transport work (mass) [g CO₂ / m tonnes · n miles]'].combine_first(dfraw['CO₂ emissions per transport work (freight) [g CO₂ / m tonnes · n miles]'])
print(f"Combined count: {dfraw['best_emission'].notna().sum()}")
