from backend.core.vessel_lookup import lookup_vessel_efficiency, load_mrv_dataset

print("Loading...")
load_mrv_dataset()
print("Lookup: MSC FORTUNATE V.025N")
res = lookup_vessel_efficiency(
    vessel_name="MSC FORTUNATE V.025N",
    carrier_name=None,
    voyage_number="025N",
    cargo_type=None
)
print("MSC FORTUNATE V.025N ->", res)

print("\nLookup: MSC FORTUNATE")
res2 = lookup_vessel_efficiency(
    vessel_name="MSC FORTUNATE",
    carrier_name=None,
    voyage_number="025N",
    cargo_type=None
)
print("MSC FORTUNATE ->", res2)
