"""Converts a dataset CSV of Oliveira e Souza et al. (2024) to the array file used by the analysis scripts.

Usage: python prepare_data.py <csv> <root>
  e.g. python prepare_data.py matlab_data_1year.csv ../runs/revised
       python prepare_data.py python_data_1year.csv ../runs/original
Keeps the 52 XMEAS/XMV channels as X, the STATUS label as y and the channel names as cols, and writes
<root>/data/tep_clean.npz (and creates <root>/out). The first column (index or timestamp) is dropped; the
separator (comma or semicolon) is detected from the header line."""
import os, sys
import numpy as np
import pandas as pd

src, root = sys.argv[1], sys.argv[2]
with open(src) as fh:
    sep = ";" if ";" in fh.readline() else ","
df = pd.read_csv(src, sep=sep, index_col=0)
cols = [c for c in df.columns if c.startswith(("XMEAS", "XMV"))]
os.makedirs(os.path.join(root, "data"), exist_ok=True); os.makedirs(os.path.join(root, "out"), exist_ok=True)
np.savez(os.path.join(root, "data", "tep_clean.npz"), X=df[cols].to_numpy(float), y=df["STATUS"].to_numpy(int), cols=np.array(cols))
print(f"{src}: {len(df)} samples, {len(cols)} channels -> {root}/data/tep_clean.npz")
