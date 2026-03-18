import os
os.environ.setdefault('GROQ_API_KEY', 'x')
os.environ.setdefault('SUPABASE_URL', 'x')
os.environ.setdefault('SUPABASE_KEY', 'x')

from backend.core.rag import query_transport_factor, query_packaging_factor
from backend.models.schemas import TransportMode, PackagingMaterial, DisposalMethod

print("=== TRANSPORT ===")
for mode in [TransportMode.SEA, TransportMode.AIR, TransportMode.TRUCK, TransportMode.RAIL]:
    f = query_transport_factor(mode)
    print(f"  {mode.value:6} -> {f.co2_per_ton_mile} | {f.source}")

print("\n=== PACKAGING ===")
for mat, disp in [
    (PackagingMaterial.CARTON, DisposalMethod.RECYCLED),
    (PackagingMaterial.CARTON, DisposalMethod.LANDFILLED),
    (PackagingMaterial.MIXED_PLASTICS, DisposalMethod.LANDFILLED),
]:
    f = query_packaging_factor(mat, disp)
    print(f"  {mat.value:15}/{disp.value:10} -> {f.co2e_per_ton} | {f.source}")