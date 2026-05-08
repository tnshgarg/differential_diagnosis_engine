import json
from collections import Counter
from pathlib import Path


TRAIN_DIR = Path("dataset/train")
EVAL_DIR = Path("dataset/eval")
EXPECTED_DAYS = [0, 5, 9, 14]
EXPECTED_ACTIONS = {"observe", "refer_PHC", "defer"}
EXPECTED_NEW_DISTRIBUTION = {2: 35, 1: 10, -1: 5}
PREFIXES = ["TB", "DENGUE", "TYPHOID"]


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_file(path):
    errors = []
    data = load_json(path)

    for key in ["disease_ground_truth", "pivot_visit", "visits"]:
        if key not in data:
            errors.append(f"missing {key}")

    visits = data.get("visits", [])
    if len(visits) != 4:
        errors.append(f"expected 4 visits, got {len(visits)}")
        return errors

    for index, visit in enumerate(visits):
        if visit.get("day") != EXPECTED_DAYS[index]:
            errors.append(f"visit {index}: wrong day {visit.get('day')}")

        if "inputs" not in visit:
            errors.append(f"visit {index}: missing inputs")

        output = visit.get("expected_output", {})
        hypotheses = output.get("hypotheses", [])
        if len(hypotheses) != 3:
            errors.append(f"visit {index}: expected 3 hypotheses, got {len(hypotheses)}")
        else:
            ranks = sorted(hypothesis.get("rank") for hypothesis in hypotheses)
            if ranks != [1, 2, 3]:
                errors.append(f"visit {index}: ranks are {ranks}")

        if "pivot" not in output:
            errors.append(f"visit {index}: missing pivot")

        if output.get("action") not in EXPECTED_ACTIONS:
            errors.append(f"visit {index}: invalid action {output.get('action')}")

    return errors


def summarize_split(folder):
    counts = Counter()
    for path in folder.glob("*.json"):
        prefix = path.stem.rsplit("_", 1)[0]
        counts[prefix] += 1
    return counts


def verify_new_distribution():
    errors = []
    for prefix in PREFIXES:
        files = sorted(TRAIN_DIR.glob(f"{prefix}_*.json"))
        new_files = [
            path for path in files if 21 <= int(path.stem.rsplit("_", 1)[1]) <= 70
        ]
        if len(new_files) != 50:
            errors.append(f"{prefix}: expected 50 new trajectories, got {len(new_files)}")
            continue

        pivot_counts = Counter(load_json(path).get("pivot_visit") for path in new_files)
        for pivot_visit, expected in EXPECTED_NEW_DISTRIBUTION.items():
            if pivot_counts[pivot_visit] != expected:
                errors.append(
                    f"{prefix}: pivot_visit {pivot_visit} expected {expected}, got {pivot_counts[pivot_visit]}"
                )

    return errors


def main():
    all_errors = []
    for folder in [TRAIN_DIR, EVAL_DIR]:
        for path in sorted(folder.glob("*.json")):
            errors = validate_file(path)
            for error in errors:
                all_errors.append(f"{path}: {error}")

    all_errors.extend(verify_new_distribution())

    train_counts = summarize_split(TRAIN_DIR)
    eval_counts = summarize_split(EVAL_DIR)

    print("Train counts:", dict(sorted(train_counts.items())))
    print("Eval counts:", dict(sorted(eval_counts.items())))
    print("Train total:", sum(train_counts.values()))
    print("Eval total:", sum(eval_counts.values()))

    if all_errors:
        print("\nVerification failed:")
        for error in all_errors:
            print("-", error)
        raise SystemExit(1)

    print("\nVerification passed.")


if __name__ == "__main__":
    main()
