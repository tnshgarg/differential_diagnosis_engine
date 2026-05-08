"""
Arogya Radar — LLM-Powered Dataset Augmentation Pipeline
Uses Groq API with llama-3.3-70b-versatile to generate diverse clinical
reasoning trajectories for TB, Dengue, and Typhoid.

Schema follows mega_prompt.md exactly.
"""

import json
import os
import re
import time
import random
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
MODEL              = "llama-3.3-70b-versatile"
TRAIN_DIR          = "dataset/train"
EVAL_DIR           = "dataset/eval"
TARGET_TOTAL_PER_DISEASE = 120   # total trajectories desired per disease
SLEEP_BETWEEN      = 1.5  # seconds between calls

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)

# ── PERSONAS ──────────────────────────────────────────────────────────────────
PERSONAS = [
    "28-saal ki mahila, khet mein kaam karti hai",
    "45-saal ka mazdoor aadmi, daily wage",
    "8-saal ka kamzor ladka",
    "60-saal ki akeli budhiya, diabetes bhi hai",
    "35-saal ki pregnant aurat",
    "22-saal ka hostel mein rehne wala college student",
    "55-saal ka kisan, beedi peeta hai",
    "12-saal ki school ladki",
    "40-saal ka auto driver, dhoop mein rehta hai",
    "70-saal ka buzurg, joint pain rehta hai",
    "30-saal ki 2 bachon ki maa",
    "18-saal ka ITI student",
    "50-saal ki aurat, BP ki dawai leti hai",
    "5-saal ka bachha, bahut rota hai",
    "38-saal ka truck driver",
    "25-saal ki nursing student",
    "48-saal ka dukandaar",
    "15-saal ka cricketer ladka",
    "33-saal ki anganwadi worker",
    "65-saal ka retired teacher",
]

IMG_DAY9 = [
    "visible weight loss, sunken cheeks",
    "pale appearance, dark circles under eyes",
    "red rash on arms and chest",
    "swollen lymph nodes on neck",
    "patient looks exhausted, lying down",
    "sweating visible on forehead",
    "petechiae rash spots on torso",
    "prescription from previous PHC visit shown",
    "patient holding stomach in pain",
    "bleeding from gums visible",
]

IMG_DAY14 = [
    "blood-tinged sputum on cloth",
    "severe weakness, cannot stand",
    "prescription and blood test report shown",
    "petechiae on lower limbs",
    "jaundiced appearance",
    "dehydrated, dry cracked lips",
    "holding chest while coughing",
    "distended abdomen",
    "bruising on arms without injury",
    "patient on cot, unable to move",
]

# ── DISEASE CONFIG ────────────────────────────────────────────────────────────
DISEASES = {
    "Tuberculosis": {
        "confuser1": "Bronchitis",
        "confuser2": "Viral fever",
        "confuser3": "Pneumonia",
        "d0_hint": "mild fever and fatigue — generic viral presentation",
        "d5_hint": "persistent dry cough begins — could be bronchitis",
        "d9_hint": "CARDINAL: cough >2 weeks, raat ko pasina (night sweats), wazan kam hona (weight loss)",
        "d14_hint": "severe weakness, possible balgam mein khoon (blood in sputum)",
        "discriminating": "raat ko pasina, wazan gir raha hai, khaansi 2 hafte se zyada",
    },
    "Dengue": {
        "confuser1": "Viral fever",
        "confuser2": "Chikungunya",
        "confuser3": "Malaria",
        "d0_hint": "sudden very high fever, severe headache — acute onset",
        "d5_hint": "aankhon ke peeche dard (retro-orbital pain), haddiyan toot rahi hain (bone pain)",
        "d9_hint": "CARDINAL: fever drops, lal daane / rash appears, masoodon se khoon (bleeding gums)",
        "d14_hint": "recovery or hemorrhagic signs, extreme weakness",
        "discriminating": "aankhon ke peeche dard, lal daane, masoodon se khoon",
    },
    "Typhoid": {
        "confuser1": "Gastroenteritis",
        "confuser2": "Viral fever",
        "confuser3": "Malaria",
        "d0_hint": "low-grade fever, malaise, headache — very generic",
        "d5_hint": "bukhar roz badh raha hai (step-wise fever), pet dard, kabz (constipation)",
        "d9_hint": "CARDINAL: tez bukhar, kabz ke baad dast (diarrhea after constipation), severe weakness",
        "d14_hint": "exhaustion, continuous high fever, risk of intestinal perforation",
        "discriminating": "bukhar roz badh raha hai, kabz ke baad dast, pet mein bahut dard",
    },
}

# ── BUILD PROMPT ──────────────────────────────────────────────────────────────
def build_prompt(disease, persona, img9, img14, traj_type):
    cfg = DISEASES[disease]

    if traj_type == "typical":
        type_note = "TYPICAL: clear progression, pivot_visit=2 (day 9 is the pivot visit)."
        pivot_instruction = "pivot_visit should be 2"
        d9_pivot = "true"
        d9_action = "refer_PHC"
    elif traj_type == "variant":
        type_note = "VARIANT: pivot happens earlier at visit index 1 (day 5). pivot_visit=1."
        pivot_instruction = "pivot_visit should be 1"
        d9_pivot = "false"
        d9_action = "refer_PHC"
    else:  # atypical
        type_note = "ATYPICAL: symptoms never fully clear, model stays uncertain, action=defer throughout. pivot_visit=-1."
        pivot_instruction = "pivot_visit should be -1"
        d9_pivot = "false"
        d9_action = "defer"

    prompt = f"""You generate clinical training data for Arogya Radar — an AI that helps rural Indian ASHA workers diagnose diseases.

Generate 1 patient trajectory JSON for: {disease}
Type: {type_note}
Patient: {persona}

Day-by-day clinical guide:
- Day 0: {cfg['d0_hint']}
- Day 5: {cfg['d5_hint']}
- Day 9: {cfg['d9_hint']}
- Day 14: {cfg['d14_hint']}
Key discriminating symptoms: {cfg['discriminating']}

Return ONLY valid JSON — no markdown, no extra text. Follow this EXACT schema:

{{
  "disease_ground_truth": "{disease}",
  "pivot_visit": <number — {pivot_instruction}>,
  "visits": [
    {{
      "day": 0,
      "inputs": {{
        "text": "<Hinglish — what patient says on day 0, vary from generic fever/fatigue/ache>",
        "image_desc": null
      }},
      "expected_output": {{
        "hypotheses": [
          {{"label": "{cfg['confuser2']}", "rank": 1, "rationale": "<clinical reason>"}},
          {{"label": "{cfg['confuser1']}", "rank": 2, "rationale": "<clinical reason>"}},
          {{"label": "{disease}", "rank": 3, "rationale": "<clinical reason>"}}
        ],
        "discriminating_symptoms": [],
        "pivot": false,
        "action": "observe"
      }}
    }},
    {{
      "day": 5,
      "inputs": {{
        "text": "<Hinglish — day 5 new symptoms, progression>",
        "image_desc": null
      }},
      "expected_output": {{
        "hypotheses": [
          {{"label": "<rank1 label>", "rank": 1, "rationale": "<reason>"}},
          {{"label": "<rank2 label>", "rank": 2, "rationale": "<reason>"}},
          {{"label": "<rank3 label>", "rank": 3, "rationale": "<reason>"}}
        ],
        "discriminating_symptoms": ["<Hindi symptom that shifted ranking>"],
        "pivot": false,
        "action": "observe"
      }}
    }},
    {{
      "day": 9,
      "inputs": {{
        "text": "<Hinglish — cardinal signs appearing>",
        "image_desc": "{img9}"
      }},
      "expected_output": {{
        "hypotheses": [
          {{"label": "{disease}", "rank": 1, "rationale": "<reason cardinal signs confirm {disease}>"}},
          {{"label": "{cfg['confuser1']}", "rank": 2, "rationale": "<reason>"}},
          {{"label": "{cfg['confuser3']}", "rank": 3, "rationale": "<reason>"}}
        ],
        "discriminating_symptoms": ["<key discriminating symptoms in Hindi>"],
        "pivot": {d9_pivot},
        "action": "{d9_action}"
      }}
    }},
    {{
      "day": 14,
      "inputs": {{
        "text": "<Hinglish — confirmed/worsening>",
        "image_desc": "{img14}"
      }},
      "expected_output": {{
        "hypotheses": [
          {{"label": "{disease}", "rank": 1, "rationale": "<confirmation reason>"}},
          {{"label": "{cfg['confuser1']}", "rank": 2, "rationale": "Ruled out"}},
          {{"label": "{cfg['confuser3']}", "rank": 3, "rationale": "Ruled out"}}
        ],
        "discriminating_symptoms": ["<confirming symptom>"],
        "pivot": false,
        "action": "refer_PHC"
      }}
    }}
  ]
}}"""
    return prompt


# ── VALIDATE ──────────────────────────────────────────────────────────────────
def validate(traj):
    errors = []
    for key in ["disease_ground_truth", "pivot_visit", "visits"]:
        if key not in traj:
            errors.append(f"missing {key}")
    visits = traj.get("visits", [])
    if len(visits) != 4:
        errors.append(f"expected 4 visits, got {len(visits)}")
        return errors
    expected_days = [0, 5, 9, 14]
    for i, v in enumerate(visits):
        if v.get("day") != expected_days[i]:
            errors.append(f"visit {i}: wrong day {v.get('day')}")
        hyps = v.get("expected_output", {}).get("hypotheses", [])
        if len(hyps) != 3:
            errors.append(f"visit {i}: need 3 hypotheses, got {len(hyps)}")
        if "inputs" not in v:
            errors.append(f"visit {i}: missing inputs")
    return errors


# ── PARSE RESPONSE ────────────────────────────────────────────────────────────
def parse(text):
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        data = json.loads(text)
        return [data] if isinstance(data, dict) else data
    except json.JSONDecodeError:
        # Try to find a JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return [json.loads(match.group())]
            except Exception:
                pass
    return []


# ── MAIN ──────────────────────────────────────────────────────────────────────
def count_existing(prefix):
    total = 0
    for d in [TRAIN_DIR, EVAL_DIR]:
        total += sum(1 for f in os.listdir(d) if f.startswith(prefix + "_"))
    return total


def main():
    diseases = ["Tuberculosis", "Dengue", "Typhoid"]
    prefixes = {"Tuberculosis": "TB", "Dengue": "DENGUE", "Typhoid": "TYPHOID"}

    total_generated = 0
    total_errors    = 0

    # Distribution for 50 new per disease:
    # 35 typical, 10 variant, 5 atypical
    type_schedule = (
        ["typical"] * 35 +
        ["variant"] * 10 +
        ["atypical"] * 5
    )

    for disease in diseases:
        prefix   = prefixes[disease]
        start_idx = count_existing(prefix) + 1
        remaining = TARGET_TOTAL_PER_DISEASE - count_existing(prefix)
        
        if remaining <= 0:
            print(f"  ✓ {disease} already has {count_existing(prefix)} trajectories. Skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"GENERATING {remaining} TRAJECTORIES FOR {disease.upper()}")
        print(f"Starting at index {start_idx}")
        print(f"{'='*55}")

        batch_num = 0

        while remaining > 0:
            traj_type = random.choice(["typical", "typical", "typical", "variant", "atypical"])
            persona   = random.choice(PERSONAS)
            img9      = random.choice(IMG_DAY9)
            img14     = random.choice(IMG_DAY14)

            prompt = build_prompt(disease, persona, img9, img14, traj_type)

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical dataset generator. Output ONLY valid JSON. No markdown. No explanation."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.85,
                    max_tokens=2000,
                )

                raw  = response.choices[0].message.content
                data = parse(raw)

                if not data:
                    print(f"  [{start_idx}] PARSE FAIL — skipping")
                    total_errors += 1
                    time.sleep(SLEEP_BETWEEN)
                    continue

                traj = data[0]
                errs = validate(traj)
                if errs:
                    print(f"  [{start_idx}] VALIDATION ERRORS: {errs}")
                    total_errors += 1
                    time.sleep(SLEEP_BETWEEN)
                    continue

                traj["disease_ground_truth"] = disease  # enforce correct label

                label    = f"{prefix}_{start_idx:03d}"
                out_path = os.path.join(TRAIN_DIR, f"{label}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(traj, f, indent=2, ensure_ascii=False)

                print(f"  ✓ {label} ({traj_type}) — pivot_visit={traj.get('pivot_visit')}")
                start_idx      += 1
                remaining      -= 1
                total_generated += 1

            except Exception as e:
                print(f"  API ERROR: {e}")
                total_errors += 1
                time.sleep(5)
                continue

            batch_num += 1
            time.sleep(SLEEP_BETWEEN)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    train_count = len(os.listdir(TRAIN_DIR))
    eval_count  = len(os.listdir(EVAL_DIR))

    print(f"\n{'='*55}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*55}")
    print(f"New trajectories generated : {total_generated}")
    print(f"Errors / skipped           : {total_errors}")
    print(f"Train set total            : {train_count}")
    print(f"Eval set total             : {eval_count}")
    print(f"Grand total                : {train_count + eval_count}")


if __name__ == "__main__":
    main()
