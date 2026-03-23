import backend.core.vessel_lookup as vl
vl.load_mrv_dataset()
from rapidfuzz import fuzz, process
names = vl._df['name_lower'].to_list()
top = process.extract('msc gulsun', names, limit=5, scorer=fuzz.token_sort_ratio)
for match_name, ratio, idx in top:
    row = vl._df.iloc[idx]
    comp_col = 'Company name' if 'Company name' in row else 'Verifier Name'
    db_carrier = str(row.get(comp_col, '')).lower()
    partial = fuzz.partial_ratio('msc', db_carrier)
    carrier_score = partial * 0.35
    ns = ratio * 0.40
    total = ns + carrier_score + 10.0 + 5.0
    vessel = row['Name']
    print(f"Vessel={vessel} | ratio={ratio:.1f} | company={repr(db_carrier[:40])} | partial={partial} | cs={carrier_score:.1f} | total={total:.2f}")
