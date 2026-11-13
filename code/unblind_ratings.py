"""
unblind_ratings.py
After rating is finished: put the participant IDs back on the ratings saved from the dashboard.
Nothing happens automatically. You run this yourself when you are done. If some ratings are
missing it tells you which and asks whether to go ahead.

Run:  python unblind_ratings.py path/to/physio_qc_ratings.csv
Writes physio_qc_ratings_unblinded.csv next to the ratings file.
"""
import sys
from pathlib import Path
import pandas as pd
import process_physio as pp

ratings_file = Path(sys.argv[1])
ratings = pd.read_csv(ratings_file, dtype=str)
key = pd.read_csv(pp.ROOT / "blinding_key.csv")
merged = key.merge(ratings, left_on="code", right_on="recording", how="left").drop(columns="recording")

unrated = merged.loc[merged[["heart_rate", "breathing"]].isna().any(axis=1), "code"]
if len(unrated):
    print(f"{len(unrated)} recording(s) have a missing rating: {', '.join(unrated)}")
    if input("Do you really want to unblind now? Type y and press Enter: ").strip().lower() != "y":
        sys.exit("Stopped. Nothing was unblinded.")

merged = merged.sort_values("participant")
merged.to_csv(ratings_file.with_name("physio_qc_ratings_unblinded.csv"), index=False)
print(merged.to_string(index=False))
