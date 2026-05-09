import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def sorted_hypotheses(hypotheses: List[Dict]) -> List[Dict]:
    return sorted(hypotheses, key=lambda h: h.get("rank", 99))


def format_visit_timeline(visit: Dict) -> str:
    day = visit.get("day")
    inputs = visit.get("inputs", {})
    text = inputs.get("text", "").strip()
    image_desc = inputs.get("image_desc")
    image_text = image_desc if image_desc else "none"
    return (
        f"- Day {day}\n"
        f"  - Patient statement: {text}\n"
        f"  - Visual clue: {image_text}"
    )


def build_user_prompt(trajectory_id: str, trajectory: Dict) -> str:
    visits = trajectory.get("visits", [])
    visit_blocks = [format_visit_timeline(v) for v in visits]
    timeline = "\n".join(visit_blocks)

    return (
        "You are assisting an ASHA worker with longitudinal differential diagnosis.\n"
        "Analyze the following patient timeline across visits and provide clinical reasoning.\n\n"
        f"Trajectory ID: {trajectory_id}\n"
        "Visit timeline:\n"
        f"{timeline}\n\n"
        "Please summarize:\n"
        "1) How the differential diagnosis evolved over time,\n"
        "2) Which discriminating symptoms changed ranking,\n"
        "3) When/why pivot happened (or why it never did),\n"
        "4) Why the final action is appropriate."
    )


def summarize_visit_reasoning(index: int, visit: Dict) -> Tuple[str, bool]:
    output = visit.get("expected_output", {})
    hypotheses = sorted_hypotheses(output.get("hypotheses", []))
    if len(hypotheses) >= 3:
        top = hypotheses[0]
        second = hypotheses[1]
        third = hypotheses[2]
        ranking_text = (
            f"{top.get('label')} (rank 1), {second.get('label')} (rank 2), "
            f"{third.get('label')} (rank 3)"
        )
    else:
        ranking_text = "ranking unavailable"

    discr = output.get("discriminating_symptoms", [])
    discr_text = ", ".join(discr) if discr else "no new discriminating symptom yet"
    action = output.get("action", "unknown")
    pivot = bool(output.get("pivot", False))
    day = visit.get("day")

    line = (
        f"Visit {index + 1} (Day {day}): Differential is {ranking_text}. "
        f"Key signal: {discr_text}. Action chosen: {action}."
    )
    if pivot:
        line += " This is the pivot visit because the leading diagnosis becomes clinically convincing."
    return line, pivot


def build_assistant_summary(trajectory_id: str, trajectory: Dict) -> str:
    visits = trajectory.get("visits", [])
    pivot_visit = trajectory.get("pivot_visit")
    disease = trajectory.get("disease_ground_truth", "Unknown")

    visit_lines = []
    pivot_flags = []
    final_action = "unknown"

    for idx, visit in enumerate(visits):
        line, is_pivot = summarize_visit_reasoning(idx, visit)
        visit_lines.append(line)
        pivot_flags.append(is_pivot)
        final_action = visit.get("expected_output", {}).get("action", final_action)

    pivot_indices = [i for i, p in enumerate(pivot_flags) if p]
    if pivot_indices:
        observed_pivot_text = ", ".join(str(i) for i in pivot_indices)
        pivot_explanation = (
            f"Observed pivot visit index from labels: {observed_pivot_text}. "
            f"Declared pivot_visit is {pivot_visit}."
        )
    else:
        pivot_explanation = (
            f"No explicit pivot flag was triggered in visits. "
            f"Declared pivot_visit is {pivot_visit}, consistent with atypical/uncertain progression."
        )

    return (
        f"Clinical reasoning summary for {trajectory_id}:\n"
        f"- Ground truth trajectory pattern is most consistent with {disease}.\n"
        + "\n".join(f"- {line}" for line in visit_lines)
        + "\n"
        + f"- {pivot_explanation}\n"
        + f"- Final management decision: {final_action}. This aligns with the final risk profile and symptom trajectory."
    )


def convert_file(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trajectory_id = path.stem
    user_content = build_user_prompt(trajectory_id, data)
    assistant_content = build_assistant_summary(trajectory_id, data)

    return {
        "id": trajectory_id,
        "messages": [
            { "role": "system",
             "content": "You are an AI medical reasoning assistant helping ASHA workers analyze longitudinal disease progression safely and cautiously."
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert trajectory JSONs into Unsloth SFTTrainer chat-format JSONL."
    )
    parser.add_argument(
        "--input_dir",
        default="dataset/train",
        help="Directory containing trajectory JSON files.",
    )
    parser.add_argument(
        "--output_file",
        default="train.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in: {input_dir}")

    records = [convert_file(path) for path in files]

    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} trajectories from {input_dir} -> {output_file}")


if __name__ == "__main__":
    main()
