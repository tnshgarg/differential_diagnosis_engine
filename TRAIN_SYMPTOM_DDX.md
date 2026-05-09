# Symptom DDX Fine-Tune Runbook

This is the second fine-tune. It replaces the old timeline-summary experiment for the app task:

```text
patient symptoms -> structured differential diagnosis JSON
```

## Files

- `symptom_ddx_train.jsonl` - training data for symptom-to-DDX JSON
- `prepare_symptom_ddx_dataset.py` - deterministic generator for the JSONL
- `validate_symptom_ddx_dataset.py` - validates JSONL and assistant JSON completions
- `train.py` - Unsloth SFT script, now defaulting to `symptom_ddx_train.jsonl`

## Local Checks

```bash
python3 prepare_symptom_ddx_dataset.py
python3 validate_symptom_ddx_dataset.py
python3 -m py_compile train.py prepare_symptom_ddx_dataset.py validate_symptom_ddx_dataset.py
```

## Colab Training

Use a T4 GPU runtime.

```python
!pip install -q unsloth transformers accelerate bitsandbytes datasets trl
```

Restart runtime:

```python
import os
os.kill(os.getpid(), 9)
```

Upload these files into Colab:

```text
train.py
symptom_ddx_train.jsonl
```

Train:

```python
!python train.py
```

The adapter will be saved to:

```text
outputs_symptom_ddx_plain_v3
```

## Upload Adapter To Hugging Face

Use a new Hugging Face write token.

```python
from huggingface_hub import login, whoami, HfApi

login()

username = whoami()["name"]
repo_id = f"{username}/gemma4-ddx-symptom-json"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    folder_path="outputs_symptom_ddx_plain_v3",
    repo_id=repo_id,
    repo_type="model",
)

print("Uploaded to:", repo_id)
```

## Inference Test

```python
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs_symptom_ddx_plain_v3",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="gemma")
FastLanguageModel.for_inference(model)
```

```python
prompt = """Patient symptoms: fever, cough, chest pain, shortness of breath.
Return differential diagnosis JSON."""

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=500,
    do_sample=False,
    repetition_penalty=1.05,
    pad_token_id=tokenizer.eos_token_id,
)

answer = outputs[0][inputs["input_ids"].shape[-1]:]
text = tokenizer.decode(answer, skip_special_tokens=True)
text = text.replace("<end_of_turn>", "").strip()
print(text)
```

Expected output: raw JSON with keys:

```text
differentials, red_flags, next_questions, next_tests, missing_information, disclaimer
```
