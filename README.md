# Arogya Radar Differential Diagnosis Engine

Dataset preparation pipeline for temporal differential diagnosis across:
- Tuberculosis
- Dengue
- Typhoid

The goal is to train/evaluate a model that updates ranked hypotheses across repeated patient visits (day 0 -> day 5 -> day 9 -> day 14), similar to how ASHA workers observe progression in rural settings.

## What This Project Contains

- A curated trajectory dataset in JSON format (`dataset/train`, `dataset/eval`)
- Scripted generation of initial seed trajectories (`generate_dataset.py`)
- Validation and train/eval split from seed data (`validate_and_split.py`)
- LLM-based augmentation for scaling the dataset (`augment_dataset.py`)
- Resume utility for filling missing generated IDs (`resume_remaining_dataset.py`)
- End-to-end dataset verification (`verify_dataset.py`)
- Raw DDXPlus filtering helper used during early preparation (`filter_ddxplus.py`)

## Current Dataset Snapshot

Based on `python3 verify_dataset.py` in this repo:

- Train: `198` files
  - `TB`: 68
  - `DENGUE`: 67
  - `TYPHOID`: 63
- Eval: `12` files
  - `TB`: 2
  - `DENGUE`: 3
  - `TYPHOID`: 7

## Trajectory Schema (Per Patient)

Each JSON trajectory has:

- `disease_ground_truth`: one of Tuberculosis, Dengue, Typhoid
- `pivot_visit`: visit index where diagnosis becomes decisive (`2`, `1`, or `-1` for atypical)
- `visits`: exactly 4 visits at days `[0, 5, 9, 14]`

Each visit includes:

- `inputs.text`: patient complaint in Hinglish
- `inputs.image_desc`: `null` or short visual cue text
- `expected_output.hypotheses`: exactly 3 ranked hypotheses (`rank` 1,2,3)
- `expected_output.discriminating_symptoms`: symptoms that changed ranking
- `expected_output.pivot`: `true` only when that visit is decisive
- `expected_output.action`: one of `observe`, `refer_PHC`, `defer`

## End-to-End Workflow (What Was Done)

1. Disease profile design  
   `disease_specs.md` documents cardinal signs, progression, confusers, and action thresholds.

2. Initial trajectory generation (60 samples)  
   `generate_dataset.py` produced `dataset_60_trajectories.txt` with labeled blocks like `=== TB_001 ===`.

3. Parse, validate, split seed data  
   `validate_and_split.py` validates required fields and writes JSON files into:
   - `dataset/train/`
   - `dataset/eval/`

4. Scale dataset with LLM augmentation  
   `augment_dataset.py` uses Groq (`llama-3.3-70b-versatile`) to generate additional trajectories, with target distribution for new IDs (`021-070`) per disease:
   - 35 typical (`pivot_visit = 2`)
   - 10 variant (`pivot_visit = 1`)
   - 5 atypical (`pivot_visit = -1`)

5. Resume incomplete runs (if needed)  
   `resume_remaining_dataset.py` generates only missing IDs while preserving existing files.

6. Final verification  
   `verify_dataset.py` checks schema consistency, rank validity, action validity, counts, and new-ID pivot distribution.

## Repository Structure

- `dataset/train/` - training trajectories
- `dataset/eval/` - held-out evaluation trajectories
- `generate_dataset.py` - creates initial 60 trajectory text bundle
- `validate_and_split.py` - parses and writes seed JSON files
- `augment_dataset.py` - Groq-based augmentation generator
- `resume_remaining_dataset.py` - resume only missing generated IDs
- `verify_dataset.py` - integrity checks for entire dataset
- `filter_ddxplus.py` - filters DDXPlus source to target diseases
- `filtered_tb_dengue_typhoid.csv` - filtered DDXPlus subset
- `disease_specs.md` - clinical progression guide used for generation
- `mega_prompt.md` - original structured prompt used to bootstrap generation
- `requirements.txt` - Python dependencies

## Setup

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment (only needed for augmentation scripts)

Create `.env` in repo root:

```env
GROQ_API_KEY=your_api_key_here
```

## Common Commands

### Verify existing dataset

```bash
python3 verify_dataset.py
```

### Rebuild seed split from `dataset_60_trajectories.txt`

```bash
python3 validate_and_split.py
```

### Generate new augmented trajectories

```bash
python3 augment_dataset.py
```

### Fill only missing IDs in augmented range

```bash
python3 resume_remaining_dataset.py
```

## Notes and Caveats

- Augmentation scripts call a live LLM API and may require retries/rate-limit waits.
- Generated Hinglish text is intentionally varied for realism.
- `verify_dataset.py` is the single source of truth for final integrity before training.
- Keep `dataset/train` and `dataset/eval` under version control to preserve reproducibility.
