# -*- coding: utf-8 -*-
"""
Untitled0 (Colab workflow) — symptom DDX LoRA train + sanity inference + optional Hub upload.

Original notebook:
  https://colab.research.google.com/drive/17XNfuWqYUtz10ldenvpvYeNJPMLqnlAG

How to use in Google Colab
--------------------------
1. Hugging Face auth (pick one — **no Colab Secrets required**):

   - **A (easiest in Colab):** run once per runtime, in a cell **before** training::

         from huggingface_hub import login
         login()

   - **B:** set a runtime-only env var in a cell (token does **not** stay in the notebook file)::

         import os
         os.environ["HF_TOKEN"] = "hf_..."   # paste once; delete this cell after if you prefer

   - **C:** this script can prompt for a hidden token (`getpass`) if you pass ``--prompt-hf-token``.

2. After first-time `pip install` below completes: **Runtime → Restart session**, then **Run all**.
   (This replaces the old `os.kill`-style reboot.)

3. Upload `train.py` and `symptom_ddx_train.jsonl` to the Colab cwd (typically `/content`),
   OR mount Drive / clone this repo.

4. Keep **MAX_SEQ_LENGTH=2048** (see `train.py`) so JSON completions are not truncated.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# --- Colab Cell 1: dependencies (runs as normal Python subprocess) -------------
def pip_install_deps() -> None:
    deps = (
        "unsloth",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "trl",
        "huggingface_hub",
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *deps])


def _hf_token_already_usable() -> bool:
    """True if downloads/uploads can use HuggingFace auth without a new login."""
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import HfFolder

        return bool(HfFolder.get_token())
    except Exception:
        return False


def ensure_hf_authenticated(*, prompt_token: bool = False) -> None:
    """
    Hugging Face auth without Colab Secrets.

    Order: existing ``HF_TOKEN`` or saved token → try interactive ``login()`` →
    if ``--prompt-hf-token``, ask once with hidden ``getpass``.
    """
    if _hf_token_already_usable():
        return

    try:
        from huggingface_hub import login

        login(add_to_git_credential=False)
    except Exception:
        pass

    if _hf_token_already_usable():
        return

    if prompt_token:
        try:
            from getpass import getpass

            token = getpass(
                "HF token (hidden; create at https://huggingface.co/settings/tokens): "
            ).strip()
            if token:
                os.environ["HF_TOKEN"] = token
        except Exception:
            pass

    if not _hf_token_already_usable():
        print(
            "HF auth not found. In Colab, run this once before training:\n"
            "  from huggingface_hub import login; login()\n"
            "Or set for this session only: os.environ['HF_TOKEN'] = 'hf_...'\n"
            "Or run: python untitled0.py --prompt-hf-token"
        )


def maybe_rename_browser_uploads() -> None:
    """If uploads land as 'train (2).py', normalize names expected by repo."""
    renames = [
        ("train (2).py", "train.py"),
        ("symptom_ddx_train (2).jsonl", "symptom_ddx_train.jsonl"),
    ]
    for src_name, dst_name in renames:
        src = Path(src_name)
        dst = Path(dst_name)
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))


def try_import_training_stack() -> bool:
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import trl  # noqa: F401
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def prompt_install_or_restart_if_needed(skip_pip: bool) -> None:
    if skip_pip:
        return
    if try_import_training_stack():
        return
    print("Installing dependencies …")
    pip_install_deps()
    print(
        "Install finished. On Colab choose: Runtime → Restart session, "
        "then run this notebook again WITHOUT re-installing deps."
    )
    raise SystemExit(0)


def run_training(
    *,
    dataset_path: str,
    output_dir: str,
    max_seq_length: int,
    epochs: float,
    run_post_infer: bool,
) -> None:
    env = os.environ.copy()
    env["DATASET_PATH"] = dataset_path
    env["OUTPUT_DIR"] = output_dir
    env["MAX_SEQ_LENGTH"] = str(max_seq_length)
    env["EPOCHS"] = str(epochs)
    env["RUN_POST_TRAIN_INFERENCE"] = "1" if run_post_infer else "0"

    subprocess.check_call([sys.executable, "train.py"], env=env, cwd=os.getcwd())


def inference_apply_chat_template(
    adapter_dir: str,
    *,
    max_seq_length: int,
    user_prompt: str,
) -> str:
    import torch

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(f"Missing unsloth. Install deps first. ({exc})") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    tensors = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    pad_id = tokenizer.pad_token_id
    attention_mask = tensors.ne(pad_id) if pad_id is not None else None

    with torch.no_grad():
        generated = model.generate(
            input_ids=tensors,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )

    continuation = generated[0][tensors.shape[-1] :]
    text = tokenizer.decode(continuation, skip_special_tokens=False)
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>", 1)[0].strip()
    return text


def upload_folder_to_hf(
    *,
    folder_path: str,
    repo_id_env: str | None,
    private_repo: bool,
) -> None:
    from huggingface_hub import HfApi, whoami

    ensure_hf_authenticated(prompt_token=False)
    if not _hf_token_already_usable():
        print("Skipping Hub upload — run login() first or set HF_TOKEN for this session.")
        return

    repo_id = repo_id_env or f"{whoami()['name']}/gemma4-ddx-symptom-json"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private_repo, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=folder_path)
    print("Uploaded adapters to Hub:", repo_id)


def zip_outputs(folder_path: Path) -> Path | None:
    """
    Create ``<folder_name>_bundle.zip`` beside the directory (Colab-friendly download).
    Returns path to the zip or None if folder is missing.
    """
    if not folder_path.is_dir():
        return None
    archive_base = folder_path.parent / f"{folder_path.name}_bundle"
    created = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=str(folder_path.parent),
        base_dir=folder_path.name,
    )
    print("Zipped adapters to:", created)
    return Path(created)


def maybe_download_zip_colab(zip_file: Path | None) -> None:
    try:
        from google.colab import files  # type: ignore
    except Exception:
        print("Skipping download helper — only available in Google Colab.")
        return

    if zip_file is not None and zip_file.exists():
        files.download(str(zip_file))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab-style driver for symptom DDX training.")
    parser.add_argument("--skip-pip-bootstrap", action="store_true")
    parser.add_argument("--dataset-path", default="symptom_ddx_train.jsonl")
    parser.add_argument("--output-dir", default="outputs_symptom_ddx")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--train-smoke-inside-train-py",
        action="store_true",
        help="If set, forwards RUN_POST_TRAIN_INFERENCE=1 to train.py (skip if you prefer one Colab inference only).",
    )
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--upload-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default=None)
    parser.add_argument("--public-hub", action="store_true")
    parser.add_argument(
        "--infer-prompt",
        default=(
            "Patient symptoms: fever, cough, breathing difficulty.\n"
            "Return differential diagnosis JSON only."
        ),
    )
    parser.add_argument(
        "--prompt-hf-token",
        action="store_true",
        help="If not already logged in, ask for HF token once via hidden prompt (getpass).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    prompt_install_or_restart_if_needed(skip_pip=args.skip_pip_bootstrap)
    ensure_hf_authenticated(prompt_token=args.prompt_hf_token)
    maybe_rename_browser_uploads()

    if args.infer_only:
        summary = inference_apply_chat_template(
            args.output_dir,
            max_seq_length=args.max_seq_length,
            user_prompt=args.infer_prompt,
        )
        print("\n=== Inference (apply_chat_template) ===\n")
        print(summary)
        return

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise SystemExit(
            f"Missing dataset at {dataset_path.resolve()}. "
            "Upload symptom_ddx_train.jsonl next to train.py."
        )

    train_py = Path("train.py")
    if not train_py.exists():
        raise SystemExit("Missing train.py in current directory — upload clone or copy from repo.")

    run_training(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        run_post_infer=args.train_smoke_inside_train_py,
    )

    outp = Path(args.output_dir)
    if outp.exists():
        zip_file = zip_outputs(outp)
        maybe_download_zip_colab(zip_file)

    if args.upload_hub:
        upload_folder_to_hf(
            folder_path=args.output_dir,
            repo_id_env=args.hub_repo_id,
            private_repo=not args.public_hub,
        )

    demo = inference_apply_chat_template(
        args.output_dir,
        max_seq_length=args.max_seq_length,
        user_prompt=args.infer_prompt,
    )
    print("\n=== Manual inference sanity (same tokenizer path as train) ===\n")
    print(demo)


if __name__ == "__main__":
    main(sys.argv[1:])
