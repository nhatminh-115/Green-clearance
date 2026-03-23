import json
from backend.core.vessel_lookup import lookup_vessel_efficiency, load_mrv_dataset

load_mrv_dataset()
res = lookup_vessel_efficiency(
    vessel_name="MSC FORTUNATE V.025N",
    carrier_name=None,
    voyage_number="025N",
    cargo_type=None
)
res2 = lookup_vessel_efficiency(
    vessel_name="MSC FORTUNATE",
    carrier_name=None,
    voyage_number="025N",
    cargo_type=None
)

output = {
    "V.025N": res.model_dump(),
    "Clean": res2.model_dump()
}
with open("test_out.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
