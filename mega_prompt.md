# AROGYA RADAR — DATASET PREPARATION MEGA PROMPT
# Paste this entire prompt into a new Claude conversation

---

You are helping me build a clinical reasoning training dataset for an AI 
system called Arogya Radar. This system helps rural ASHA workers in India 
diagnose TB, Dengue, and Typhoid early by tracking patient symptoms across 
multiple visits over time.

I need you to complete 3 steps in order. Do not skip any step. Do not move 
to the next step until the current one is fully complete.

=============================================================
BACKGROUND CONTEXT (read this before starting)
=============================================================

The final goal is 60 JSON files that look like this exact schema:

{
  "disease_ground_truth": "Tuberculosis",
  "pivot_visit": 3,
  "visits": [
    {
      "day": 0,
      "inputs": {
        "text": "Bukhar hai, sar dard, thakaan",
        "image_desc": null
      },
      "expected_output": {
        "hypotheses": [
          {"label": "Viral fever",  "rank": 1, "rationale": "Common fever presentation day 0"},
          {"label": "Dengue-like",  "rank": 2, "rationale": "Cannot rule out yet"},
          {"label": "TB-like",      "rank": 3, "rationale": "No cardinal signs yet"}
        ],
        "discriminating_symptoms": [],
        "pivot": false,
        "action": "observe"
      }
    },
    {
      "day": 5,
      "inputs": {
        "text": "Bukhar abhi bhi hai, thodi khaansi bhi shuru hui",
        "image_desc": null
      },
      "expected_output": {
        "hypotheses": [
          {"label": "Viral fever",  "rank": 1, "rationale": "Still most likely at day 5"},
          {"label": "TB-like",      "rank": 2, "rationale": "Cough developing, watch"},
          {"label": "Dengue-like",  "rank": 3, "rationale": "No rash or platelet drop"}
        ],
        "discriminating_symptoms": ["khaansi"],
        "pivot": false,
        "action": "observe"
      }
    },
    {
      "day": 9,
      "inputs": {
        "text": "Khaansi badh gayi, raat ko pasina aata hai, weight kam ho raha hai",
        "image_desc": "visible weight loss, pale appearance"
      },
      "expected_output": {
        "hypotheses": [
          {"label": "TB-like",      "rank": 1, "rationale": "Night sweats + weight loss + cough past day 7 is TB cardinal"},
          {"label": "Viral fever",  "rank": 2, "rationale": "Fever pattern atypical for simple viral"},
          {"label": "Dengue-like",  "rank": 3, "rationale": "No rash, no platelet drop"}
        ],
        "discriminating_symptoms": ["raat ko pasina", "weight loss", "khaansi past day 7"],
        "pivot": true,
        "action": "refer_PHC"
      }
    },
    {
      "day": 14,
      "inputs": {
        "text": "Bahut kamzori, khaansi mein thoda khoon bhi aaya",
        "image_desc": "prescription from previous visit shown"
      },
      "expected_output": {
        "hypotheses": [
          {"label": "TB-like",      "rank": 1, "rationale": "Haemoptysis confirms TB trajectory"},
          {"label": "Viral fever",  "rank": 2, "rationale": "Ruled mostly out"},
          {"label": "Dengue-like",  "rank": 3, "rationale": "Ruled out"}
        ],
        "discriminating_symptoms": ["khoon aana khaansi mein"],
        "pivot": false,
        "action": "refer_PHC"
      }
    }
  ]
}

RULES FOR ALL 60 TRAJECTORIES:
- Every trajectory has exactly 4 visits at day 0, day 5, day 9, day 14
- Text inputs must be in Hindi/Hinglish (the way a rural patient actually speaks)
- image_desc is null for visits 1 and 2, add a description for visits 3 and 4
- hypotheses always has exactly 3 entries ranked 1, 2, 3
- pivot: true only at the visit where the ground truth disease reaches rank 1
- pivot_visit number matches the visit index (0-based) where pivot becomes true
- action is one of: "observe", "refer_PHC", "defer"
- defer means the model is not confident enough to commit to any hypothesis

=============================================================
STEP 1 — BUILD 3 DISEASE PROFILE SPECS
=============================================================

Before generating any JSON, first write a detailed disease profile spec 
for each of the 3 diseases. This is a structured reference document that 
you will use in Step 2 to generate accurate trajectories.

For EACH disease (Tuberculosis, Dengue, Typhoid) write:

DISEASE: [name]

A. CARDINAL SIGNS
   List the 3-4 symptoms that almost always appear and are specific to 
   this disease. These are the symptoms that should trigger pivot: true.

B. DAY-BY-DAY PROGRESSION
   Day 0  : What the patient feels on day 1. Usually ambiguous.
   Day 3  : What new symptoms appear. Still looks like something else.
   Day 5  : Progression. Starting to differentiate.
   Day 7  : Cardinal signs begin appearing.
   Day 9  : Cardinal signs clear. This is usually the pivot visit.
   Day 14 : Confirmed trajectory. Action required.

C. DISCRIMINATING SYMPTOMS
   Symptoms that, when they appear, should shift the hypothesis ranking 
   toward this disease specifically. Include the Hindi/Hinglish name for 
   each symptom.

D. RED HERRINGS
   Symptoms this disease shares with the other 2 diseases that could 
   cause confusion. Explain why each one is misleading.

E. TYPICAL MISDIAGNOSIS
   Which disease does this get confused with most and at which day? 
   Why does the confusion happen?

F. CO-PRESENTATION RISK
   Which other disease can appear simultaneously? What makes a 
   co-presentation trajectory different from a single disease trajectory?

G. ASHA ACTION THRESHOLD
   At exactly what symptom combination should the ASHA worker refer 
   the patient to PHC vs continue observing?

Write all 3 specs completely before moving to Step 2.

=============================================================
STEP 2 — GENERATE 60 TRAJECTORY JSONs
=============================================================

Using the 3 disease specs you just wrote in Step 1, generate the 
following trajectories:

TUBERCULOSIS: 20 trajectories
  - 14 typical (clear TB progression, pivot at visit 3)
  - 4 variant (different symptom orderings, pivot at visit 2 or 3)
  - 2 atypical (symptoms never fully clear, action = "defer")

DENGUE: 20 trajectories
  - 13 typical (clear dengue progression, pivot at visit 2 or 3)
  - 4 variant (different symptom orderings)
  - 2 atypical (action = "defer")
  - 1 co-presentation with Typhoid (both diseases present simultaneously, 
    hypotheses show both competing, action = "defer" or "refer_PHC")

TYPHOID: 20 trajectories
  - 14 typical (clear typhoid progression, pivot at visit 3)
  - 4 variant (different symptom orderings)
  - 2 atypical (action = "defer")

VARIATION RULES (so trajectories are not all identical):
  - Vary the day the cardinal sign first appears (sometimes day 7, 
    sometimes day 9)
  - Vary the initial complaint (some start with headache, some with 
    fever, some with body ache)
  - Vary the image_desc (sometimes rash photo, sometimes weight loss, 
    sometimes prescription photo)
  - Some patients are children (age in rationale), some are adults
  - Some trajectories have a wrong first hypothesis that gets corrected 
    by visit 2

OUTPUT FORMAT:
  Output each trajectory as a properly formatted JSON object.
  Label each one clearly before the JSON like:
  
  === TB_001 ===
  { ... json ... }
  
  === TB_002 ===
  { ... json ... }
  
  Continue until all 60 are done.
  Do not stop early. Generate all 60 completely.

=============================================================
STEP 3 — GIVE ME THE SPLIT SUMMARY
=============================================================

After generating all 60 trajectories, give me a summary table like this:

TRAIN SET (48 trajectories):
  TB typical      : 14
  TB variant      : 4
  Dengue typical  : 13
  Dengue variant  : 4
  Typhoid typical : 14 (some moved to eval)
  Typhoid variant : 3  (some moved to eval)
  Total           : 48

EVAL SET (12 trajectories):
  TB atypical        : 1
  Dengue atypical    : 1
  Typhoid atypical   : 1
  Co-presentation    : 1
  Typical (mixed)    : 8
  Total              : 12

Then tell me exactly which trajectory labels (TB_001, DENGUE_014 etc) 
go into train and which go into eval.

=============================================================
START NOW WITH STEP 1
=============================================================

Begin with the 3 disease profile specs. Be medically accurate. 
Use WHO guidelines knowledge for day-by-day progression. 
Do not rush. A good spec in Step 1 means accurate JSONs in Step 2.
