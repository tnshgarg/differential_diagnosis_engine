import json
from pathlib import Path


DATASET_PATH = Path("symptom_ddx_train.jsonl")
REQUIRED_RESPONSE_KEYS = {
    "differentials",
    "red_flags",
    "next_questions",
    "next_tests",
    "missing_information",
    "disclaimer",
}


def main() -> None:
    errors = []
    count = 0

    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing dataset: {DATASET_PATH}")

    with DATASET_PATH.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSONL record: {exc}")
                continue

            prompt = record.get("prompt")
            completion = record.get("completion")
            if not isinstance(prompt, str) or not prompt:
                errors.append(f"line {line_number}: prompt must be a non-empty string")
                continue
            if not isinstance(completion, str) or not completion:
                errors.append(f"line {line_number}: completion must be a non-empty string")
                continue

            assistant_text = completion
            try:
                response = json.loads(assistant_text)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: assistant text is not valid JSON: {exc}")
                continue

            missing = REQUIRED_RESPONSE_KEYS - set(response)
            if missing:
                errors.append(f"line {line_number}: response missing keys {sorted(missing)}")

            differentials = response.get("differentials")
            if not isinstance(differentials, list) or not differentials:
                errors.append(f"line {line_number}: differentials must be a non-empty list")
            elif len(differentials) > 5:
                errors.append(f"line {line_number}: expected at most 5 differentials")

            for phrase in ["confirmed", "definitely", "ruled out"]:
                if phrase in assistant_text.lower():
                    errors.append(f"line {line_number}: unsafe certainty phrase found: {phrase}")

    print(f"Checked {count} records from {DATASET_PATH}")
    if errors:
        print("\nValidation failed:")
        for error in errors[:50]:
            print("-", error)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors")
        raise SystemExit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
