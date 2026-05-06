import json
import pandas as pd
import ast

# ── STEP 1: Read conditions.json and find your 3 disease names ────────────────
print("STEP 1: Reading conditions.json...")

with open("release_conditions.json") as f:
    conditions = json.load(f)

print(f"Total diseases in DDXPlus: {len(conditions)}")

# ── STEP 2: Search for TB, Dengue, Typhoid and store their exact names ─────────
print("\nSTEP 2: Finding your 3 target diseases...")

TARGET_KEYWORDS = ["tuberculosis", "dengue", "typhoid"]
TARGET_NAMES = []  # this list gets filled automatically

for key, val in conditions.items():
    name = val.get("condition_name") or val.get("cond-name-eng") or key
    if any(kw in name.lower() for kw in TARGET_KEYWORDS):
        TARGET_NAMES.append(name)
        print(f"  FOUND --> {name}  (ICD-10: {val.get('icd10-id', '—')})")

print(f"\nWill filter train.csv for: {TARGET_NAMES}")

# ── STEP 3: Filter train.csv and save as new CSV ───────────────────────────────
print("\nSTEP 3: Filtering train (1).csv ... (takes 1-2 mins, file is 671MB)")

chunks = []
for chunk in pd.read_csv("train (1).csv", chunksize=50_000):
    filtered = chunk[chunk["PATHOLOGY"].isin(TARGET_NAMES)]
    if len(filtered):
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)

print(f"Done. Rows found per disease:")
print(df["PATHOLOGY"].value_counts().to_string())

# ── STEP 4: Show sample differential rankings then save ────────────────────────
print("\nSTEP 4: Sample differential diagnosis per disease:")

for disease in df["PATHOLOGY"].unique():
    print(f"\n  --- {disease} ---")
    sample = df[df["PATHOLOGY"] == disease].head(3)
    for _, row in sample.iterrows():
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        ranked = sorted(ddx, key=lambda x: -x[1])
        for dname, prob in ranked[:5]:
            print(f"    {dname:<40} {prob:.3f}")

# ── SAVE ───────────────────────────────────────────────────────────────────────
df.to_csv("filtered_tb_dengue_typhoid.csv", index=False)

print("\n" + "="*50)
print("DONE. Saved: filtered_tb_dengue_typhoid.csv")
print(f"Total rows: {len(df)}")
print("This file is your input for building disease profile specs.")
print("="*50)
