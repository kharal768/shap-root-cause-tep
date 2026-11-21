"""Best fixed (fault-agnostic) channel list of size 3-5 against the documented channels: the largest number of
evaluated faults whose documented channels a single list can contain.

Usage: python fixed_list_bound.py. Paper: Section 4.2, Supplementary Section S12."""
import itertools
from fault_metadata import GROUND_TRUTH_CHANNELS as GT

channels = sorted({c for v in GT.values() for c in v})
print(f"{len(GT)} faults, {len(channels)} distinct documented channels")
for k in (3, 4, 5):
    best = max(itertools.combinations(channels, k), key=lambda c: sum(any(x in c for x in GT[f]) for f in GT))
    print(f"{k} channels: {sum(any(x in best for x in GT[f]) for f in GT)} of {len(GT)} faults", best)
