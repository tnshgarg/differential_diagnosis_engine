"""
Resume only the missing screenshot-target dataset items.

Target from the screenshot:
- 50 new trajectories per disease, IDs 021-070
- Distribution per disease: 35 typical, 10 variant, 5 atypical

This script leaves existing files untouched and writes only missing JSON files.
"""

import json
import os
import random
import re
import time

from dotenv import load_dotenv
from groq import Groq

import augment_dataset as aug

load_dotenv()

TRAIN_DIR = aug.TRAIN_DIR
MODEL = aug.MODEL
SLEEP_BETWEEN = aug.SLEEP_BETWEEN
TARGET_RANGE = range(21, 71)

PREFIXES = {
    "Dengue": "DENGUE",
    "Typhoid": "TYPHOID",
}

TARGET_TYPES = {
    "typical": 35,
    "variant": 10,
    "atypical": 5,
}


def existing_new_items(prefix):
    items = []
    for path in sorted(os.listdir(TRAIN_DIR)):
        if not path.startswith(prefix + "_") or not path.endswith(".json"):
            continue
        idx = int(path.removesuffix(".json").split("_")[1])
        if idx not in TARGET_RANGE:
            continue
        full_path = os.path.join(TRAIN_DIR, path)
        with open(full_path, encoding="utf-8") as f:
            traj = json.load(f)
        pivot = traj.get("pivot_visit")
        if pivot == 2:
            traj_type = "typical"
        elif pivot == 1:
            traj_type = "variant"
        elif pivot == -1:
            traj_type = "atypical"
        else:
            traj_type = "unknown"
        items.append((idx, traj_type))
    return items


def remaining_schedule(prefix):
    existing = existing_new_items(prefix)
    have = {name: 0 for name in TARGET_TYPES}
    for _, traj_type in existing:
        if traj_type in have:
            have[traj_type] += 1

    schedule = []
    for traj_type, target in TARGET_TYPES.items():
        schedule.extend([traj_type] * max(0, target - have[traj_type]))
    return schedule


def missing_indices(prefix):
    present = {idx for idx, _ in existing_new_items(prefix)}
    return [idx for idx in TARGET_RANGE if idx not in present]


def generate_one(client, disease, idx, traj_type):
    persona = random.choice(aug.PERSONAS)
    img9 = random.choice(aug.IMG_DAY9)
    img14 = random.choice(aug.IMG_DAY14)
    prompt = aug.build_prompt(disease, persona, img9, img14, traj_type)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a medical dataset generator. Output ONLY valid JSON. No markdown. No explanation.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.85,
        max_tokens=2000,
    )

    data = aug.parse(response.choices[0].message.content)
    if not data:
        raise ValueError("parse failed")

    traj = data[0]
    errors = aug.validate(traj)
    if errors:
        raise ValueError(f"validation failed: {errors}")

    traj["disease_ground_truth"] = disease
    return traj


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)
    generated = 0
    errors = 0

    for disease, prefix in PREFIXES.items():
        schedule = remaining_schedule(prefix)
        indices = missing_indices(prefix)
        if len(schedule) != len(indices):
            print(
                f"{prefix}: distribution needs {len(schedule)} items but {len(indices)} IDs are missing; "
                "using the shorter list."
            )

        print(f"\n{prefix}: generating {min(len(schedule), len(indices))} missing items")
        for idx, traj_type in zip(indices, schedule):
            label = f"{prefix}_{idx:03d}"
            out_path = os.path.join(TRAIN_DIR, f"{label}.json")
            if os.path.exists(out_path):
                continue

            while True:
                try:
                    traj = generate_one(client, disease, idx, traj_type)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(traj, f, indent=2, ensure_ascii=False)
                    print(f"  ✓ {label} ({traj_type})")
                    generated += 1
                    break
                except Exception as exc:
                    errors += 1
                    message = str(exc)
                    wait_match = re.search(r"try again in ([0-9.]+)m([0-9.]*)s", message)
                    if wait_match:
                        minutes = float(wait_match.group(1))
                        seconds = float(wait_match.group(2) or 0)
                        wait_seconds = int(minutes * 60 + seconds + 15)
                        print(
                            f"  ! {label} ({traj_type}) rate-limited; waiting {wait_seconds}s before retry"
                        )
                        time.sleep(wait_seconds)
                    else:
                        print(f"  ! {label} ({traj_type}) failed: {exc}; retrying")
                        time.sleep(5)

            time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. Generated={generated}, retries/errors={errors}")


if __name__ == "__main__":
    main()
