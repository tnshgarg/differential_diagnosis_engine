import json
import random

# Datasets arrays
tb_trajectories = []
dengue_trajectories = []
typhoid_trajectories = []

def create_trajectory(disease, prefix, index, typicality, pivot_visit, visits_data):
    return {
        "id": f"{prefix}_{index:03d}",
        "disease_ground_truth": disease,
        "type": typicality,
        "pivot_visit": pivot_visit,
        "visits": visits_data
    }

# --- TB Generators ---
def gen_tb_typical(index):
    return create_trajectory("Tuberculosis", "TB", index, "typical", 2, [
        {"day": 0, "inputs": {"text": "Bukhar hai, sar dard, thakaan aur ajeeb si khasi", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Viral fever", "rank": 1, "rationale": "Common fever presentation"}, {"label": "Bronchitis", "rank": 2, "rationale": "Mild cough"}, {"label": "TB-like", "rank": 3, "rationale": "No cardinal signs"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Bukhar abhi bhi hai, khaansi thodi badh gayi hai", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Bronchitis", "rank": 1, "rationale": "Cough persists"}, {"label": "TB-like", "rank": 2, "rationale": "Cough continuing"}, {"label": "Viral fever", "rank": 3, "rationale": "Unlikely"}], "discriminating_symptoms": ["khaansi"], "pivot": False, "action": "observe"}},
        {"day": 9, "inputs": {"text": "Khaansi 2 hafte se zyada ho gayi, raat ko pasina aata hai, wazan kam ho raha hai", "image_desc": "visible weight loss"}, "expected_output": {"hypotheses": [{"label": "TB-like", "rank": 1, "rationale": "Night sweats + weight loss + prolonged cough"}, {"label": "Bronchitis", "rank": 2, "rationale": "Unlikely with weight loss"}, {"label": "Pneumonia", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": ["raat ko pasina", "wazan kam hona", "khaansi > 2 weeks"], "pivot": True, "action": "refer_PHC"}},
        {"day": 14, "inputs": {"text": "Bahut kamzori, khaansi mein thoda khoon bhi aaya", "image_desc": "blood in sputum"}, "expected_output": {"hypotheses": [{"label": "TB-like", "rank": 1, "rationale": "Haemoptysis confirms TB"}, {"label": "Pneumonia", "rank": 2, "rationale": "Ruled out"}, {"label": "Bronchitis", "rank": 3, "rationale": "Ruled out"}], "discriminating_symptoms": ["khoon aana"], "pivot": False, "action": "refer_PHC"}}
    ])

def gen_tb_variant(index):
    return create_trajectory("Tuberculosis", "TB", index, "variant", 1, [
        {"day": 0, "inputs": {"text": "Sookhi khaansi aur halka bukhar pichle 10 din se", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Bronchitis", "rank": 1, "rationale": "Prolonged cough"}, {"label": "TB-like", "rank": 2, "rationale": "Need to monitor"}, {"label": "Viral fever", "rank": 3, "rationale": "Too long for viral"}], "discriminating_symptoms": ["10 din se khaansi"], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Raat ko pasina aata hai aur khaansi mein balgam hai", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "TB-like", "rank": 1, "rationale": "Night sweats + >2 wk cough"}, {"label": "Pneumonia", "rank": 2, "rationale": "Productive cough"}, {"label": "Bronchitis", "rank": 3, "rationale": "Unlikely"}], "discriminating_symptoms": ["raat ko pasina"], "pivot": True, "action": "refer_PHC"}},
        {"day": 9, "inputs": {"text": "Wazan kam lag raha hai, saans lene mein takleef", "image_desc": "pale appearance"}, "expected_output": {"hypotheses": [{"label": "TB-like", "rank": 1, "rationale": "Consistent TB"}, {"label": "Pneumonia", "rank": 2, "rationale": "Less likely"}, {"label": "Asthma", "rank": 3, "rationale": "Wheezing absent"}], "discriminating_symptoms": ["wazan kam hona"], "pivot": False, "action": "refer_PHC"}},
        {"day": 14, "inputs": {"text": "Condition worsening, khoon aaya", "image_desc": "prescription shown"}, "expected_output": {"hypotheses": [{"label": "TB-like", "rank": 1, "rationale": "Confirmed"}, {"label": "Pneumonia", "rank": 2, "rationale": "No"}, {"label": "Bronchitis", "rank": 3, "rationale": "No"}], "discriminating_symptoms": ["khoon aaya"], "pivot": False, "action": "refer_PHC"}}
    ])

def gen_tb_atypical(index):
    return create_trajectory("Tuberculosis", "TB", index, "atypical", -1, [
        {"day": 0, "inputs": {"text": "Pet dard aur halka bukhar", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Food poisoning", "rank": 1, "rationale": "Common"}, {"label": "Typhoid-like", "rank": 2, "rationale": "Possible"}, {"label": "Viral fever", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Pet dard thik hai par thakaan bahut hai aur raat ko halka pasina", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Viral fever", "rank": 1, "rationale": "Fatigue"}, {"label": "TB-like", "rank": 2, "rationale": "Night sweats"}, {"label": "Typhoid-like", "rank": 3, "rationale": "Fever"}], "discriminating_symptoms": ["raat ko pasina"], "pivot": False, "action": "defer"}},
        {"day": 9, "inputs": {"text": "Sookhi khaansi shuru hui", "image_desc": "looks tired"}, "expected_output": {"hypotheses": [{"label": "TB-like", "rank": 1, "rationale": "Cough + sweats"}, {"label": "Bronchitis", "rank": 2, "rationale": "Cough"}, {"label": "Viral fever", "rank": 3, "rationale": "Extended"}], "discriminating_symptoms": ["khaansi"], "pivot": False, "action": "defer"}},
        {"day": 14, "inputs": {"text": "Khasi aur thakaan", "image_desc": "no clear visible signs"}, "expected_output": {"hypotheses": [{"label": "TB-like", "rank": 1, "rationale": "Prolonged"}, {"label": "Bronchitis", "rank": 2, "rationale": "Possible"}, {"label": "Viral fever", "rank": 3, "rationale": "No"}], "discriminating_symptoms": [], "pivot": False, "action": "defer"}}
    ])

# Generate TB
for i in range(1, 15): tb_trajectories.append(gen_tb_typical(i))
for i in range(15, 19): tb_trajectories.append(gen_tb_variant(i))
for i in range(19, 21): tb_trajectories.append(gen_tb_atypical(i))

# --- Dengue Generators ---
def gen_dengue_typical(index):
    return create_trajectory("Dengue", "DENGUE", index, "typical", 2, [
        {"day": 0, "inputs": {"text": "Achanak se tez bukhar aur sar dard shuru hua", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Viral fever", "rank": 1, "rationale": "Sudden fever"}, {"label": "Dengue-like", "rank": 2, "rationale": "Sudden onset"}, {"label": "Malaria", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": ["achanak tez bukhar"], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Aankhon ke peeche dard aur haddiyan toot rahi hain", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Retro-orbital and joint pain"}, {"label": "Viral fever", "rank": 2, "rationale": "Unlikely severe"}, {"label": "Chikungunya", "rank": 3, "rationale": "Joint pain"}], "discriminating_symptoms": ["aankhon ke peeche dard", "haddiyon mein dard"], "pivot": False, "action": "observe"}},
        {"day": 9, "inputs": {"text": "Bukhar kam hai par lal daane nikal aaye", "image_desc": "red rash visible on arms"}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Rash after fever drop"}, {"label": "Measles", "rank": 2, "rationale": "Rash"}, {"label": "Chikungunya", "rank": 3, "rationale": "Less likely"}], "discriminating_symptoms": ["lal daane"], "pivot": True, "action": "refer_PHC"}},
        {"day": 14, "inputs": {"text": "Chakkar aa rahe hain aur masoodon se khoon aya", "image_desc": "pale, bleeding gums"}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Hemorrhagic signs"}, {"label": "Chikungunya", "rank": 2, "rationale": "Ruled out"}, {"label": "Viral fever", "rank": 3, "rationale": "Ruled out"}], "discriminating_symptoms": ["masoodon se khoon"], "pivot": False, "action": "refer_PHC"}}
    ])

def gen_dengue_variant(index):
    return create_trajectory("Dengue", "DENGUE", index, "variant", 1, [
        {"day": 0, "inputs": {"text": "Bahut jodo mein dard aur thakaan", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Chikungunya", "rank": 1, "rationale": "Joint pain"}, {"label": "Dengue-like", "rank": 2, "rationale": "Pain"}, {"label": "Viral fever", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": ["jodo mein dard"], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Tez bukhar aur lal daane", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Fever + Rash"}, {"label": "Measles", "rank": 2, "rationale": "Rash"}, {"label": "Chikungunya", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": ["lal daane", "tez bukhar"], "pivot": True, "action": "refer_PHC"}},
        {"day": 9, "inputs": {"text": "Kamzori bahut zyada hai", "image_desc": "weakness"}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Post-viral fatigue"}, {"label": "Chikungunya", "rank": 2, "rationale": "Unlikely"}, {"label": "Measles", "rank": 3, "rationale": "Unlikely"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}},
        {"day": 14, "inputs": {"text": "Ab thik lag raha hai", "image_desc": "improving"}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Recovery"}, {"label": "Chikungunya", "rank": 2, "rationale": "No"}, {"label": "Viral fever", "rank": 3, "rationale": "No"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}}
    ])

def gen_dengue_atypical(index):
    return create_trajectory("Dengue", "DENGUE", index, "atypical", -1, [
        {"day": 0, "inputs": {"text": "Halka bukhar aur sar dard", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Viral fever", "rank": 1, "rationale": "Common"}, {"label": "Dengue-like", "rank": 2, "rationale": "Possible"}, {"label": "Typhoid-like", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Bukhar nahi utar raha", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Viral fever", "rank": 1, "rationale": "Persists"}, {"label": "Typhoid-like", "rank": 2, "rationale": "Continuous"}, {"label": "Dengue-like", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": [], "pivot": False, "action": "defer"}},
        {"day": 9, "inputs": {"text": "Thodi chakkar aa rahi hai", "image_desc": "no rash"}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Dizziness"}, {"label": "Viral fever", "rank": 2, "rationale": "Fatigue"}, {"label": "Typhoid-like", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": ["chakkar"], "pivot": False, "action": "defer"}},
        {"day": 14, "inputs": {"text": "Kamzori", "image_desc": "looks weak"}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Recovery"}, {"label": "Viral fever", "rank": 2, "rationale": "Possible"}, {"label": "Typhoid-like", "rank": 3, "rationale": "No"}], "discriminating_symptoms": [], "pivot": False, "action": "defer"}}
    ])

def gen_dengue_copresentation(index):
    return create_trajectory("Dengue + Typhoid", "DENGUE", index, "co-presentation", -1, [
        {"day": 0, "inputs": {"text": "Tez bukhar aur pet mein dard", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "Abdominal"}, {"label": "Dengue-like", "rank": 2, "rationale": "High fever"}, {"label": "Food poisoning", "rank": 3, "rationale": "Pain"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Bukhar badh gaya hai, jodo mein dard bhi", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Joint pain"}, {"label": "Typhoid-like", "rank": 2, "rationale": "Fever increasing"}, {"label": "Malaria", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": ["jodo mein dard"], "pivot": False, "action": "refer_PHC"}},
        {"day": 9, "inputs": {"text": "Lal daane aur dast (diarrhea)", "image_desc": "rash visible"}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Rash"}, {"label": "Typhoid-like", "rank": 2, "rationale": "Diarrhea"}, {"label": "Viral fever", "rank": 3, "rationale": "No"}], "discriminating_symptoms": ["lal daane", "dast"], "pivot": False, "action": "refer_PHC"}},
        {"day": 14, "inputs": {"text": "Chakkar aa rahe hain", "image_desc": "pale"}, "expected_output": {"hypotheses": [{"label": "Dengue-like", "rank": 1, "rationale": "Weakness"}, {"label": "Typhoid-like", "rank": 2, "rationale": "Prolonged"}, {"label": "Viral fever", "rank": 3, "rationale": "No"}], "discriminating_symptoms": [], "pivot": False, "action": "refer_PHC"}}
    ])

for i in range(1, 14): dengue_trajectories.append(gen_dengue_typical(i))
for i in range(14, 18): dengue_trajectories.append(gen_dengue_variant(i))
for i in range(18, 20): dengue_trajectories.append(gen_dengue_atypical(i))
dengue_trajectories.append(gen_dengue_copresentation(20))

# --- Typhoid Generators ---
def gen_typhoid_typical(index):
    return create_trajectory("Typhoid", "TYPHOID", index, "typical", 2, [
        {"day": 0, "inputs": {"text": "Halka bukhar aur ajeeb si thakaan", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Viral fever", "rank": 1, "rationale": "Mild fever"}, {"label": "Typhoid-like", "rank": 2, "rationale": "Malaise"}, {"label": "Malaria", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Bukhar roz badh raha hai, pet mein dard aur kabz hai", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "Step-wise fever + constipation"}, {"label": "Gastroenteritis", "rank": 2, "rationale": "Abdominal pain"}, {"label": "Viral fever", "rank": 3, "rationale": "Less likely"}], "discriminating_symptoms": ["bukhar roz badh raha hai", "kabz"], "pivot": False, "action": "observe"}},
        {"day": 9, "inputs": {"text": "Bahut tez bukhar hai aur dast lag gaye", "image_desc": "looks exhausted"}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "High fever + diarrhea"}, {"label": "Gastroenteritis", "rank": 2, "rationale": "Diarrhea"}, {"label": "Dengue-like", "rank": 3, "rationale": "Unlikely"}], "discriminating_symptoms": ["tez bukhar", "dast"], "pivot": True, "action": "refer_PHC"}},
        {"day": 14, "inputs": {"text": "Halat kharab hai, pet mein bahut dard", "image_desc": "holding stomach in pain"}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "Complications risk"}, {"label": "Gastroenteritis", "rank": 2, "rationale": "Unlikely"}, {"label": "Malaria", "rank": 3, "rationale": "No"}], "discriminating_symptoms": ["pet mein bahut dard"], "pivot": False, "action": "refer_PHC"}}
    ])

def gen_typhoid_variant(index):
    return create_trajectory("Typhoid", "TYPHOID", index, "variant", 1, [
        {"day": 0, "inputs": {"text": "Dast aur thakaan lag rahi hai", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Gastroenteritis", "rank": 1, "rationale": "Diarrhea"}, {"label": "Food poisoning", "rank": 2, "rationale": "Diarrhea"}, {"label": "Typhoid-like", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": ["dast"], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Bukhar shuru ho gaya aur badh raha hai, pet dard hai", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "Increasing fever + abdominal"}, {"label": "Gastroenteritis", "rank": 2, "rationale": "Still possible"}, {"label": "Malaria", "rank": 3, "rationale": "No"}], "discriminating_symptoms": ["bukhar badh raha hai", "pet dard"], "pivot": True, "action": "refer_PHC"}},
        {"day": 9, "inputs": {"text": "Dast aur tez bukhar", "image_desc": "sweaty"}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "Classic signs"}, {"label": "Gastroenteritis", "rank": 2, "rationale": "No"}, {"label": "Dengue-like", "rank": 3, "rationale": "No"}], "discriminating_symptoms": ["tez bukhar"], "pivot": False, "action": "refer_PHC"}},
        {"day": 14, "inputs": {"text": "Kamzori", "image_desc": "improving"}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "Recovery"}, {"label": "Gastroenteritis", "rank": 2, "rationale": "No"}, {"label": "Viral fever", "rank": 3, "rationale": "No"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}}
    ])

def gen_typhoid_atypical(index):
    return create_trajectory("Typhoid", "TYPHOID", index, "atypical", -1, [
        {"day": 0, "inputs": {"text": "Sirf sar dard aur thakaan", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Viral fever", "rank": 1, "rationale": "Common"}, {"label": "Migraine", "rank": 2, "rationale": "Headache"}, {"label": "Typhoid-like", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": [], "pivot": False, "action": "observe"}},
        {"day": 5, "inputs": {"text": "Bukhar aana shuru hua", "image_desc": None}, "expected_output": {"hypotheses": [{"label": "Viral fever", "rank": 1, "rationale": "Fever"}, {"label": "Typhoid-like", "rank": 2, "rationale": "Fever"}, {"label": "Malaria", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": [], "pivot": False, "action": "defer"}},
        {"day": 9, "inputs": {"text": "Thoda pet kharab hai", "image_desc": "no clear signs"}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "Mild GI"}, {"label": "Viral fever", "rank": 2, "rationale": "Persists"}, {"label": "Gastroenteritis", "rank": 3, "rationale": "Possible"}], "discriminating_symptoms": ["pet kharab"], "pivot": False, "action": "defer"}},
        {"day": 14, "inputs": {"text": "Thakaan lag rahi hai bas", "image_desc": "tired"}, "expected_output": {"hypotheses": [{"label": "Typhoid-like", "rank": 1, "rationale": "Mild case"}, {"label": "Viral fever", "rank": 2, "rationale": "Post-viral"}, {"label": "Gastroenteritis", "rank": 3, "rationale": "No"}], "discriminating_symptoms": [], "pivot": False, "action": "defer"}}
    ])

for i in range(1, 15): typhoid_trajectories.append(gen_typhoid_typical(i))
for i in range(15, 19): typhoid_trajectories.append(gen_typhoid_variant(i))
for i in range(19, 21): typhoid_trajectories.append(gen_typhoid_atypical(i))

all_data = tb_trajectories + dengue_trajectories + typhoid_trajectories

with open('dataset_60_trajectories.txt', 'w') as f:
    for item in all_data:
        # Strip out custom keys for final output
        output_dict = {
            "disease_ground_truth": item["disease_ground_truth"],
            "pivot_visit": item["pivot_visit"],
            "visits": item["visits"]
        }
        f.write(f"=== {item['id']} ===\n")
        f.write(json.dumps(output_dict, indent=2))
        f.write("\n\n")

print("Generated dataset_60_trajectories.txt successfully!")
