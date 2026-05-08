# Arogya Radar Differential Diagnosis Engine

Phase 1 dataset preparation for temporal differential diagnosis across
Tuberculosis, Dengue, and Typhoid.

## Dataset

- `dataset/train/`: 198 trajectory JSON files for fine-tuning.
- `dataset/eval/`: 12 held-out trajectory JSON files for evaluation.
- Additional generated training set: 50 new trajectories per disease
  (`021`-`070`) with `35 typical + 10 variant + 5 atypical` distribution.

Each trajectory contains 4 visits (`day 0`, `day 5`, `day 9`, `day 14`) with:

- ranked differential hypotheses,
- discriminating symptoms,
- pivot flag,
- ASHA action: `observe`, `refer_PHC`, or `defer`.

## Key Files

- `disease_specs.md`: disease progression and ASHA action thresholds.
- `dataset_60_trajectories.txt`: original generated 60-trajectory source.
- `validate_and_split.py`: parses and splits the original 60 trajectories.
- `augment_dataset.py`: Groq-based augmentation script for additional trajectories.
- `resume_remaining_dataset.py`: targeted resume script for missing `021`-`070` items.
- `verify_dataset.py`: schema, count, and distribution verification.

## Verification

```bash
python3 verify_dataset.py
```

Expected result:

```text
Train total: 198
Eval total: 12
Verification passed.
```
