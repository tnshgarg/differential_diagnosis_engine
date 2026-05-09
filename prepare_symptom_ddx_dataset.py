import argparse
import json
from itertools import count
from pathlib import Path


DISCLAIMER = "Differential support only. This is not a final diagnosis."


CONDITIONS = {
    "pneumonia": {
        "name": "Pneumonia",
        "symptom_sets": [
            ["fever", "cough", "chest pain", "shortness of breath"],
            ["fever", "productive cough", "chills", "fatigue"],
            ["cough", "pleuritic chest pain", "shortness of breath", "fever"],
        ],
        "reason": "Fever with cough, chest discomfort, and breathlessness can suggest a lower respiratory infection.",
        "alternatives": ["Acute bronchitis", "Influenza or COVID-like viral illness", "Asthma exacerbation", "Tuberculosis"],
        "red_flags": ["severe shortness of breath", "low oxygen saturation", "confusion", "blue lips", "persistent chest pain"],
        "tests": ["SpO2", "respiratory rate", "chest X-ray", "CBC", "COVID/flu test"],
        "questions": ["How many days of fever?", "Is cough dry or productive?", "Any blood in sputum?", "Any oxygen reading available?"],
    },
    "acute_bronchitis": {
        "name": "Acute bronchitis",
        "symptom_sets": [
            ["cough", "mild fever", "chest discomfort", "fatigue"],
            ["dry cough", "sore throat", "low-grade fever", "wheeze"],
            ["cough", "runny nose", "chest tightness", "body ache"],
        ],
        "reason": "Cough with mild fever and chest discomfort can fit bronchial airway inflammation, often after a viral illness.",
        "alternatives": ["Pneumonia", "Influenza or COVID-like viral illness", "Asthma exacerbation", "Upper respiratory infection"],
        "red_flags": ["shortness of breath at rest", "high fever", "coughing blood", "chest pain that is severe or persistent"],
        "tests": ["temperature", "SpO2", "lung exam", "chest X-ray if red flags are present"],
        "questions": ["How long has the cough lasted?", "Any wheezing?", "Any sputum color change?", "Any breathing difficulty?"],
    },
    "asthma_exacerbation": {
        "name": "Asthma exacerbation",
        "symptom_sets": [
            ["wheezing", "shortness of breath", "chest tightness", "cough"],
            ["cough at night", "wheeze", "breathlessness", "chest tightness"],
            ["shortness of breath", "wheezing", "trigger exposure", "dry cough"],
        ],
        "reason": "Wheezing, chest tightness, cough, and breathlessness are compatible with airway narrowing.",
        "alternatives": ["Pneumonia", "Acute bronchitis", "Anxiety or panic attack", "Pulmonary embolism"],
        "red_flags": ["unable to speak full sentences", "blue lips", "silent chest", "low oxygen saturation", "drowsiness"],
        "tests": ["SpO2", "peak expiratory flow if available", "respiratory rate", "lung exam"],
        "questions": ["Any known asthma?", "Any inhaler use?", "What triggered symptoms?", "Is breathlessness worsening?"],
    },
    "tuberculosis": {
        "name": "Tuberculosis-like illness",
        "symptom_sets": [
            ["cough for more than 2 weeks", "night sweats", "weight loss", "fever"],
            ["persistent cough", "blood in sputum", "weight loss", "fatigue"],
            ["long cough", "evening fever", "night sweats", "loss of appetite"],
        ],
        "reason": "Prolonged cough with night sweats, weight loss, fever, or blood in sputum is suggestive of possible pulmonary TB.",
        "alternatives": ["Pneumonia", "Chronic bronchitis", "Lung malignancy or other chronic lung disease", "Post-viral cough"],
        "red_flags": ["coughing blood", "severe weakness", "shortness of breath", "known TB exposure", "weight loss"],
        "tests": ["sputum test for TB", "chest X-ray", "SpO2", "CBC", "HIV test if clinically appropriate"],
        "questions": ["How many weeks of cough?", "Any TB contact?", "Any night sweats?", "Any weight loss?"],
    },
    "influenza_covid": {
        "name": "Influenza or COVID-like viral illness",
        "symptom_sets": [
            ["fever", "cough", "sore throat", "body aches"],
            ["sudden fever", "headache", "cough", "fatigue"],
            ["fever", "runny nose", "cough", "loss of smell"],
        ],
        "reason": "Fever with cough, sore throat, body aches, or sudden onset can fit a viral respiratory infection.",
        "alternatives": ["Pneumonia", "Acute bronchitis", "Dengue-like illness", "Malaria"],
        "red_flags": ["shortness of breath", "low oxygen saturation", "confusion", "chest pain", "dehydration"],
        "tests": ["COVID test", "flu test if available", "temperature", "SpO2"],
        "questions": ["Any sick contacts?", "When did symptoms start?", "Any breathlessness?", "Any high-risk medical condition?"],
    },
    "dengue": {
        "name": "Dengue-like illness",
        "symptom_sets": [
            ["sudden high fever", "severe body ache", "pain behind eyes", "rash"],
            ["fever", "headache", "joint pain", "rash"],
            ["fever settling", "gum bleeding", "abdominal pain", "extreme fatigue"],
        ],
        "reason": "Sudden fever with severe body ache, retro-orbital pain, rash, or bleeding symptoms can suggest dengue in an endemic setting.",
        "alternatives": ["Malaria", "Chikungunya", "Influenza or COVID-like viral illness", "Typhoid-like illness"],
        "red_flags": ["abdominal pain", "persistent vomiting", "bleeding from gums or nose", "blood in stool or vomit", "extreme restlessness or lethargy"],
        "tests": ["CBC with platelet count", "dengue NS1 or IgM based on illness day", "hematocrit", "vitals"],
        "questions": ["Which day of fever is this?", "Any rash?", "Any bleeding?", "Any abdominal pain or vomiting?"],
    },
    "malaria": {
        "name": "Malaria",
        "symptom_sets": [
            ["fever with chills", "sweating", "headache", "body ache"],
            ["intermittent fever", "chills", "weakness", "nausea"],
            ["high fever", "rigors", "vomiting", "fatigue"],
        ],
        "reason": "Fever with chills, rigors, sweating, headache, and weakness can suggest malaria in an exposure setting.",
        "alternatives": ["Dengue-like illness", "Typhoid-like illness", "Influenza or COVID-like viral illness", "Sepsis or serious infection"],
        "red_flags": ["confusion", "repeated vomiting", "seizure", "jaundice", "severe weakness"],
        "tests": ["malaria rapid diagnostic test", "peripheral smear", "CBC", "glucose if severe symptoms"],
        "questions": ["Any mosquito exposure?", "Is fever intermittent?", "Any travel to malaria area?", "Any confusion or jaundice?"],
    },
    "typhoid": {
        "name": "Typhoid-like illness",
        "symptom_sets": [
            ["fever for more than 3 days", "headache", "abdominal pain", "constipation"],
            ["prolonged fever", "weakness", "loss of appetite", "diarrhea"],
            ["stepwise fever", "stomach pain", "cough", "fatigue"],
        ],
        "reason": "Prolonged fever with abdominal symptoms, weakness, headache, constipation, or diarrhea can fit typhoid-like illness.",
        "alternatives": ["Malaria", "Dengue-like illness", "Gastroenteritis", "Viral fever"],
        "red_flags": ["severe abdominal pain", "blood in stool", "confusion", "persistent vomiting", "very high fever"],
        "tests": ["blood culture if available", "CBC", "temperature chart", "hydration assessment"],
        "questions": ["How many days of fever?", "Any unsafe food or water exposure?", "Constipation or diarrhea?", "Any severe abdominal pain?"],
    },
    "gastroenteritis": {
        "name": "Acute gastroenteritis",
        "symptom_sets": [
            ["diarrhea", "vomiting", "abdominal cramps", "mild fever"],
            ["loose stools", "nausea", "stomach pain", "dehydration"],
            ["vomiting", "watery diarrhea", "fever", "weakness"],
        ],
        "reason": "Diarrhea and vomiting with abdominal cramps can fit acute gastrointestinal infection or irritation.",
        "alternatives": ["Typhoid-like illness", "Food poisoning", "Appendicitis", "Dengue-like illness"],
        "red_flags": ["blood in stool", "severe dehydration", "persistent vomiting", "severe abdominal pain", "lethargy"],
        "tests": ["hydration assessment", "temperature", "stool test if blood or prolonged illness", "electrolytes if severe dehydration"],
        "questions": ["Any blood in stool?", "How many stools per day?", "Can the patient drink fluids?", "Any severe localized pain?"],
    },
    "appendicitis": {
        "name": "Appendicitis or acute abdomen",
        "symptom_sets": [
            ["right lower abdominal pain", "fever", "nausea", "loss of appetite"],
            ["abdominal pain moving to right side", "vomiting", "fever", "tenderness"],
            ["severe belly pain", "fever", "nausea", "worse pain with movement"],
        ],
        "reason": "Fever with localized or worsening abdominal pain, nausea, and appetite loss can suggest an acute abdomen such as appendicitis.",
        "alternatives": ["Gastroenteritis", "Urinary tract infection", "Typhoid-like illness", "Gynecologic emergency if applicable"],
        "red_flags": ["severe or worsening abdominal pain", "rigid abdomen", "fainting", "persistent vomiting", "pregnancy with abdominal pain"],
        "tests": ["abdominal exam", "CBC", "urine test", "pregnancy test if applicable", "ultrasound or urgent referral"],
        "questions": ["Where exactly is the pain?", "Did pain move location?", "Any vomiting?", "Any pregnancy possibility?"],
    },
    "uti_pyelonephritis": {
        "name": "Urinary tract infection or pyelonephritis",
        "symptom_sets": [
            ["burning urination", "frequent urination", "lower abdominal pain", "fever"],
            ["fever", "flank pain", "burning urine", "nausea"],
            ["urinary urgency", "pelvic pain", "fever", "back pain"],
        ],
        "reason": "Urinary burning, frequency, pelvic pain, fever, or flank pain can suggest urinary infection; fever/flank pain raises concern for kidney involvement.",
        "alternatives": ["Kidney stone", "Pelvic infection if applicable", "Gastroenteritis", "Appendicitis"],
        "red_flags": ["high fever with flank pain", "pregnancy", "confusion", "vomiting unable to keep fluids", "severe back pain"],
        "tests": ["urine routine/microscopy", "urine culture if available", "temperature", "hydration assessment"],
        "questions": ["Any burning while passing urine?", "Any flank pain?", "Any pregnancy?", "Any blood in urine?"],
    },
    "migraine": {
        "name": "Migraine or primary headache",
        "symptom_sets": [
            ["headache", "nausea", "light sensitivity", "sound sensitivity"],
            ["one-sided headache", "vomiting", "blurred vision", "light sensitivity"],
            ["recurrent headache", "nausea", "worse with activity", "normal alertness"],
        ],
        "reason": "Recurrent or one-sided headache with nausea and light sensitivity can fit migraine when dangerous signs are absent.",
        "alternatives": ["Meningitis", "Sinusitis", "Tension headache", "Stroke or neurologic emergency"],
        "red_flags": ["sudden worst headache", "neck stiffness with fever", "weakness on one side", "confusion", "new seizure"],
        "tests": ["neurologic exam", "temperature", "blood pressure", "urgent evaluation if red flags are present"],
        "questions": ["Is this the worst headache ever?", "Any fever or neck stiffness?", "Any weakness or speech trouble?", "Any prior similar headaches?"],
    },
    "meningitis": {
        "name": "Meningitis or serious central nervous system infection",
        "symptom_sets": [
            ["fever", "severe headache", "neck stiffness", "photophobia"],
            ["fever", "confusion", "neck pain", "vomiting"],
            ["headache", "rash", "fever", "drowsiness"],
        ],
        "reason": "Fever with severe headache, neck stiffness, photophobia, confusion, or drowsiness is concerning for serious CNS infection.",
        "alternatives": ["Migraine", "Viral fever", "Dengue-like illness", "Sepsis or serious infection"],
        "red_flags": ["neck stiffness", "confusion", "seizure", "drowsiness", "non-blanching rash"],
        "tests": ["urgent clinical evaluation", "vitals", "neurologic exam", "blood tests and lumbar puncture only in appropriate clinical setting"],
        "questions": ["Any neck stiffness?", "Any confusion or drowsiness?", "Any rash?", "Any seizure?"],
    },
    "stroke_tia": {
        "name": "Stroke or transient ischemic attack",
        "symptom_sets": [
            ["sudden face droop", "arm weakness", "speech difficulty"],
            ["sudden weakness on one side", "slurred speech", "dizziness"],
            ["sudden vision loss", "confusion", "one-sided numbness"],
        ],
        "reason": "Sudden focal neurologic symptoms such as face droop, one-sided weakness, speech difficulty, or vision loss are concerning for stroke/TIA.",
        "alternatives": ["Migraine with aura", "Seizure/post-ictal state", "Low blood sugar", "Bell palsy"],
        "red_flags": ["any sudden neurologic deficit", "speech difficulty", "face droop", "one-sided weakness", "loss of consciousness"],
        "tests": ["urgent emergency referral", "blood glucose", "blood pressure", "stroke assessment"],
        "questions": ["When was the patient last normal?", "Any one-sided weakness?", "Any speech problem?", "Any diabetes or low sugar symptoms?"],
    },
    "acs": {
        "name": "Acute coronary syndrome or cardiac chest pain",
        "symptom_sets": [
            ["chest pressure", "shortness of breath", "sweating", "nausea"],
            ["chest pain radiating to left arm", "breathlessness", "sweating"],
            ["central chest pain", "dizziness", "shortness of breath", "fatigue"],
        ],
        "reason": "Chest pressure with breathlessness, sweating, nausea, radiation, or dizziness can suggest possible cardiac ischemia.",
        "alternatives": ["Pneumonia", "Acid reflux", "Anxiety or panic attack", "Pulmonary embolism"],
        "red_flags": ["severe chest pain", "chest pain with sweating", "fainting", "shortness of breath", "pain radiating to arm or jaw"],
        "tests": ["urgent ECG", "vitals", "SpO2", "troponin in clinical facility"],
        "questions": ["When did chest pain start?", "Does it radiate to arm or jaw?", "Any sweating or fainting?", "Any cardiac risk factors?"],
    },
    "pulmonary_embolism": {
        "name": "Pulmonary embolism",
        "symptom_sets": [
            ["sudden shortness of breath", "pleuritic chest pain", "fast heartbeat"],
            ["chest pain", "shortness of breath", "coughing blood", "leg swelling"],
            ["sudden breathlessness", "dizziness", "chest pain", "recent immobility"],
        ],
        "reason": "Sudden breathlessness with pleuritic chest pain, fast heartbeat, coughing blood, leg swelling, or immobility can suggest pulmonary embolism.",
        "alternatives": ["Pneumonia", "Asthma exacerbation", "Acute coronary syndrome", "Panic attack"],
        "red_flags": ["sudden severe breathlessness", "coughing blood", "fainting", "low oxygen saturation", "one-sided leg swelling"],
        "tests": ["urgent referral", "SpO2", "heart rate", "D-dimer or CT pulmonary angiography in clinical setting"],
        "questions": ["Any recent surgery or immobility?", "Any leg swelling?", "Any coughing blood?", "Did breathlessness start suddenly?"],
    },
    "anxiety_panic": {
        "name": "Anxiety or panic attack",
        "symptom_sets": [
            ["palpitations", "chest tightness", "shortness of breath", "tingling"],
            ["sudden fear", "fast heartbeat", "breathlessness", "dizziness"],
            ["chest discomfort", "sweating", "trembling", "normal oxygen"],
        ],
        "reason": "Palpitations, sudden fear, tingling, dizziness, and chest tightness can occur with panic, but dangerous causes must be considered first.",
        "alternatives": ["Acute coronary syndrome", "Asthma exacerbation", "Pulmonary embolism", "Low blood sugar"],
        "red_flags": ["new severe chest pain", "fainting", "low oxygen saturation", "one-sided weakness", "pain radiating to arm or jaw"],
        "tests": ["vitals", "SpO2", "blood glucose if available", "ECG if chest pain or risk factors"],
        "questions": ["Any previous panic episodes?", "Any chest pain radiation?", "Any fainting?", "Any breathing disease or cardiac risk?"],
    },
}


PROMPT_TEMPLATES = [
    "Patient symptoms: {symptoms}.\nReturn differential diagnosis JSON.",
    "Symptoms reported: {symptoms}.\nGive cautious differential diagnosis as JSON only.",
    "ASHA visit complaint: {symptoms}.\nReturn JSON with differentials, red flags, next questions, and next tests.",
    "Given only these symptoms: {symptoms}.\nDo not invent history. Return structured differential diagnosis JSON.",
]


def make_differentials(primary_key: str) -> list[dict]:
    primary = CONDITIONS[primary_key]
    items = [
        {
            "name": primary["name"],
            "reasoning": primary["reason"],
            "urgency": "urgent evaluation if red flags are present",
        }
    ]
    for alt in primary["alternatives"][:4]:
        items.append(
            {
                "name": alt,
                "reasoning": "Possible alternative because overlapping symptoms can occur; additional history, exam, and basic tests are needed.",
                "urgency": "depends on severity and red flags",
            }
        )
    return items


def make_response(primary_key: str) -> str:
    condition = CONDITIONS[primary_key]
    response = {
        "differentials": make_differentials(primary_key),
        "red_flags": condition["red_flags"],
        "next_questions": condition["questions"],
        "next_tests": condition["tests"],
        "missing_information": [
            "duration not provided unless stated",
            "age not provided",
            "vitals not provided",
            "past history and medications not provided",
        ],
        "disclaimer": DISCLAIMER,
    }
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def make_record(example_id: str, prompt: str, response: str) -> dict:
    return {
        "id": example_id,
        "prompt": prompt,
        "completion": response,
    }


def build_records() -> list[dict]:
    records = []
    ids = count(1)
    for condition_key, condition in CONDITIONS.items():
        for symptoms in condition["symptom_sets"]:
            symptom_texts = [
                ", ".join(symptoms),
                "; ".join(symptoms),
                "\n- " + "\n- ".join(symptoms),
            ]
            for symptom_text in symptom_texts:
                for template in PROMPT_TEMPLATES:
                    example_id = f"SYMPTOM_DDX_{next(ids):04d}"
                    prompt = template.format(symptoms=symptom_text)
                    records.append(make_record(example_id, prompt, make_response(condition_key)))

    # Add harder negative-control examples that explicitly teach caution.
    cautious_cases = [
        (
            "fever and tiredness only",
            {
                "differentials": [
                    {
                        "name": "Viral fever or nonspecific febrile illness",
                        "reasoning": "Fever and tiredness are nonspecific and do not identify one disease.",
                        "urgency": "depends on duration, vitals, and red flags",
                    }
                ],
                "red_flags": ["confusion", "breathlessness", "persistent high fever", "severe dehydration", "rash with bleeding"],
                "next_questions": ["How many days of fever?", "Any cough, rash, abdominal pain, urinary symptoms, or travel?", "Any danger signs?"],
                "next_tests": ["temperature", "pulse", "blood pressure", "SpO2", "focused exam based on added symptoms"],
                "missing_information": ["most symptoms not provided", "duration not provided", "vitals not provided"],
                "disclaimer": DISCLAIMER,
            },
        ),
        (
            "chest pain only",
            {
                "differentials": [
                    {
                        "name": "Cardiac, lung, muscle, or gastrointestinal cause of chest pain",
                        "reasoning": "Chest pain alone is nonspecific and needs characterization before ranking causes.",
                        "urgency": "urgent if severe, exertional, associated with breathlessness, sweating, fainting, or radiation",
                    }
                ],
                "red_flags": ["severe chest pain", "shortness of breath", "sweating", "fainting", "pain radiating to arm or jaw"],
                "next_questions": ["When did pain start?", "Is it pressure-like or sharp?", "Any breathlessness or sweating?", "Any cardiac risk factors?"],
                "next_tests": ["vitals", "SpO2", "ECG if concerning", "clinical evaluation"],
                "missing_information": ["duration not provided", "pain character not provided", "vitals not provided"],
                "disclaimer": DISCLAIMER,
            },
        ),
        (
            "abdominal pain only",
            {
                "differentials": [
                    {
                        "name": "Undifferentiated abdominal pain",
                        "reasoning": "Abdominal pain alone has many possible causes; location, severity, fever, vomiting, stool, urinary, and pregnancy history are needed.",
                        "urgency": "urgent if severe, worsening, localized, with fever, fainting, bleeding, or pregnancy",
                    }
                ],
                "red_flags": ["severe worsening pain", "rigid abdomen", "blood in stool or vomit", "persistent vomiting", "pregnancy with pain"],
                "next_questions": ["Where is the pain?", "Any fever or vomiting?", "Any diarrhea or constipation?", "Any urinary symptoms?"],
                "next_tests": ["abdominal exam", "vitals", "urine test", "pregnancy test if applicable", "CBC if concerning"],
                "missing_information": ["pain location not provided", "duration not provided", "associated symptoms not provided"],
                "disclaimer": DISCLAIMER,
            },
        ),
    ]
    for symptoms, response in cautious_cases:
        for template in PROMPT_TEMPLATES:
            example_id = f"SYMPTOM_DDX_{next(ids):04d}"
            records.append(
                make_record(
                    example_id,
                    template.format(symptoms=symptoms),
                    json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build symptom-to-DDX JSONL for SFT.")
    parser.add_argument("--output_file", default="symptom_ddx_train.jsonl")
    args = parser.parse_args()

    records = build_records()
    output_path = Path(args.output_file)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
