import json
import os
import re
import random

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "dataset_60_trajectories.txt"
TRAIN_DIR   = "dataset/train"
EVAL_DIR    = "dataset/eval"

EVAL_LABELS = [
    "TB_019", "TB_020",
    "DENGUE_018", "DENGUE_019", "DENGUE_020",
    "TYPHOID_019", "TYPHOID_020",
    "TYPHOID_011", "TYPHOID_012", "TYPHOID_013", "TYPHOID_014", "TYPHOID_018"
]

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── STEP 1: READ THE FILE ─────────────────────────────────────────────────────
print("Reading dataset_60_trajectories.txt ...")
with open(INPUT_FILE, "r") as f:
    content = f.read()

# ── STEP 2: EXTRACT EACH TRAJECTORY BLOCK ────────────────────────────────────
# Looks for === TB_001 === then grabs the JSON after it
pattern = r'===\s*([\w_]+)\s*===\s*(\{.*?\}(?=\s*===|\s*$))'
matches = re.findall(pattern, content, re.DOTALL)

print(f"Found {len(matches)} trajectory blocks\n")

if len(matches) == 0:
    print("ERROR: Could not parse any trajectories.")
    print("The file format may be different. Printing first 500 chars:")
    print(content[:500])
    exit()

# ── STEP 3: VALIDATE + SAVE EACH TRAJECTORY ──────────────────────────────────
errors   = []
saved    = []
train    = []
eval_set = []

for label, json_str in matches:
    label = label.strip().upper()

    # parse JSON
    try:
        traj = json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        errors.append(f"JSON PARSE ERROR in {label}: {e}")
        continue

    # validate required fields
    checks = [
        ("disease_ground_truth" in traj,       "missing disease_ground_truth"),
        ("pivot_visit" in traj,                 "missing pivot_visit"),
        ("visits" in traj,                      "missing visits"),
        (len(traj.get("visits", [])) == 4,      "must have exactly 4 visits"),
    ]
    for passed, msg in checks:
        if not passed:
            errors.append(f"VALIDATION ERROR in {label}: {msg}")
            continue

    # validate each visit
    visit_ok = True
    for i, v in enumerate(traj["visits"]):
        if "inputs" not in v:
            errors.append(f"{label} visit {i}: missing inputs")
            visit_ok = False
        if "expected_output" not in v:
            errors.append(f"{label} visit {i}: missing expected_output")
            visit_ok = False
        elif len(v["expected_output"].get("hypotheses", [])) != 3:
            errors.append(f"{label} visit {i}: must have exactly 3 hypotheses")
            visit_ok = False

    if not visit_ok:
        continue

    # decide train or eval
    dest_dir = EVAL_DIR if label in EVAL_LABELS else TRAIN_DIR
    dest     = "eval"   if label in EVAL_LABELS else "train"

    # save individual file
    out_path = os.path.join(dest_dir, f"{label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)

    saved.append(label)
    if dest == "eval":
        eval_set.append(label)
    else:
        train.append(label)

# ── STEP 4: PRINT SUMMARY ─────────────────────────────────────────────────────
print("=" * 55)
print("VALIDATION SUMMARY")
print("=" * 55)
print(f"Total parsed   : {len(matches)}")
print(f"Successfully saved : {len(saved)}")
print(f"Errors             : {len(errors)}")
print()
print(f"Train set : {len(train)} trajectories → dataset/train/")
print(f"Eval set  : {len(eval_set)} trajectories  → dataset/eval/")

if errors:
    print("\nERRORS FOUND:")
    for e in errors:
        print(f"  ✗ {e}")
else:
    print("\nAll trajectories passed validation.")

print()
print("Train labels:", sorted(train))
print()
print("Eval labels :", sorted(eval_set))

print()
if len(saved) == 60 and len(errors) == 0:
    print("=" * 55)
    print("DATASET IS FULLY READY FOR PHASE 2")
    print("=" * 55)
elif len(saved) >= 55:
    print("Almost ready — fix the errors above and re-run.")
else:
    print("Several issues found — share the errors here for help.")
