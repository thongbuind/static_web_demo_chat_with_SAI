"""Regenerate demo answers with the local SAI 375M checkpoint.

Uses pinned upstream architecture, generation code, and the HF 100M tokenizer.
Results are resumable; demo JSON files are replaced only after every answer exists.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
HF_REV = "9b41bf35b808943d0d43f5092e259a25b25ff8c4"
GH_REV = "f41f20820322e86260ec37f4da3326a06a375fcf"
SOURCE = ROOT / "tmp/inference_source"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def download_sources():
    files = ["config/base.json", "config/375M.json", "src/inference/generate.py"]
    files += [f"src/model/{name}.py" for name in (
        "TransformerModel", "DecoderBlock", "GroupedQueryAttention",
        "RotaryPositionalEmbedding", "SwiGLU",
    )]
    targets = [(f"https://raw.githubusercontent.com/thongbuind/SAI_dev2/{GH_REV}/{f}",
                SOURCE / "github" / f) for f in files]
    targets.append((f"https://huggingface.co/thongbuind/SAI_100M/resolve/{HF_REV}/tokenizer.model",
                    SOURCE / "hf/tokenizer.model"))
    for url, destination in targets:
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-fsSL", "--max-time", "120", url, "-o", str(destination)], check=True)
    # Reorder only the used prefix of each KV cache, rather than copying its
    # unused capacity first. This preserves the exact cache values.
    generation_path = SOURCE / "github/src/inference/generate.py"
    source = generation_path.read_text()
    for name in ("key", "value"):
        old = f"{name}.index_select(0, sources)[:, :, :cache_len].clone()"
        new = f"{name}[:, :, :cache_len].index_select(0, sources).clone()"
        if old not in source and new not in source:
            raise RuntimeError("Upstream cache reordering code does not match the pinned version")
        source = source.replace(old, new)
    generation_path.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--limit", type=int, help="Generate only this many new answers; do not publish incomplete results")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()
    download_sources()
    import torch
    import sentencepiece as spm
    sys.path.insert(0, str(SOURCE / "github"))
    from src.model.TransformerModel import TransformerModel
    from src.model.RotaryPositionalEmbedding import RotaryPositionalEmbedding
    from src.inference.generate import GenerationConfig

    checkpoint = ROOT / "tmp/sft1_375M.pt"
    checkpoint_hash = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    config = json.loads((SOURCE / "github/config/base.json").read_text())
    config.update(json.loads((SOURCE / "github/config/375M.json").read_text()))
    tokenizer_path = SOURCE / "hf/tokenizer.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    assert tokenizer.get_piece_size() == config["vocab_size"]
    for token in ("[BOS]", "[EOS]", "<|im_start|>", "<|im_end|>"):
        assert tokenizer.id_to_piece(tokenizer.piece_to_id(token)) == token

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal is unavailable in this execution environment")
    keys = ("vocab_size", "d_model", "num_heads", "num_kv_heads", "num_layers", "ff_dim", "max_seq_len", "dropout")
    with torch.device("meta"):
        model = TransformerModel(**{key: config[key] for key in keys})
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    assert state["embed.weight"].data_ptr() == state["lm_head.weight"].data_ptr()
    model.lm_head.weight = model.embed.weight
    model.rope = RotaryPositionalEmbedding(config["d_model"] // config["num_heads"], config["max_seq_len"])
    dtype = getattr(torch, args.dtype)
    model = model.to(device=device, dtype=dtype).eval()
    # Beam scoring is small and causes many Metal synchronizations when each
    # candidate is copied separately. Transfer the full logits once per step.
    if device == "mps":
        original_prefill = model.prefill
        original_decode = model.decode_step
        def prefill(self, *args, **kwargs):
            logits, cache = original_prefill(*args, **kwargs)
            return logits.float().cpu(), cache
        def decode_step(self, *args, **kwargs):
            return original_decode(*args, **kwargs).float().cpu()
        model.prefill = MethodType(prefill, model)
        model.decode_step = MethodType(decode_step, model)
    if dtype != torch.float32:
        def init_cache(self, batch_size, max_gen_len, device):
            shape = (batch_size, self.num_kv_heads, max_gen_len, self.d_model // self.num_heads)
            return [[torch.empty(shape, device=device, dtype=dtype),
                     torch.empty(shape, device=device, dtype=dtype)] for _ in self.blocks]
        model.init_cache = MethodType(init_cache, model)
    del state
    generation = GenerationConfig()
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters on {device}", flush=True)

    history = ROOT / "chat_history"
    manifest = json.loads((history / "manifest.json").read_text())
    backup = ROOT / "tmp/history_100M"
    if not backup.exists():
        shutil.copytree(history, backup)
    sessions = {name: json.loads((backup / name).read_text()) for name in manifest["files"]}
    questions = list(dict.fromkeys(message["text"] for session in sessions.values()
                                   for message in session["messages"] if message["role"] == "user"))
    metadata = {
        "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_hash,
        "tokenizer_repository": "thongbuind/SAI_100M", "tokenizer_revision": HF_REV,
        "tokenizer_sha256": hashlib.file_digest(tokenizer_path.open("rb"), "sha256").hexdigest(),
        "source_repository": "https://github.com/thongbuind/SAI_dev2", "source_revision": GH_REV,
        "generation_config": asdict(generation), "context": "independent user prompts, no system prompt or conversation history",
        "device": device, "dtype": args.dtype, "torch_version": torch.__version__,
        "beam_scoring_device": "cpu", "beam_scoring_dtype": "float32",
        "source_patches": ["Reorder only the used KV cache prefix (same values)"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    cache_path = ROOT / f"tmp/results_375M_{args.dtype}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {"metadata": metadata, "answers": {}}
    for key in ("checkpoint_sha256", "tokenizer_sha256", "source_revision", "generation_config", "dtype", "beam_scoring_device"):
        if cache["metadata"][key] != metadata[key]:
            raise RuntimeError(f"Resume metadata differs: {key}")
    answers = cache["answers"]
    cache["metadata"] = metadata
    generated_count = 0
    started = time.monotonic()
    for index, question in enumerate(questions, 1):
        if question in answers:
            continue
        tick = time.monotonic()
        result = model.generate_response(question, tokenizer, config=generation, return_result=True)
        if not result.text.strip():
            raise RuntimeError(f"Empty output for {question!r}")
        answers[question] = asdict(result)
        write_json(cache_path, cache)
        if device == "mps":
            # Release unused Metal allocations between prompts to limit memory
            # and swap pressure during a long generation run.
            torch.mps.empty_cache()
        generated_count += 1
        print(f"[{len(answers)}/{len(questions)}] {time.monotonic()-tick:.1f}s | {question[:70]!r} => {result.text[:100]!r}", flush=True)
        if args.limit and generated_count >= args.limit:
            break
    if any(question not in answers for question in questions):
        print(f"Saved {len(answers)}/{len(questions)} answers; demo files not replaced yet.", flush=True)
        return

    replacements = 0
    for filename, session in sessions.items():
        question = None
        for message in session["messages"]:
            if message["role"] == "user":
                question = message["text"]
            elif message["role"] == "ai":
                if question is None:
                    raise RuntimeError(f"Missing user question in {filename}")
                message["text"] = answers[question]["text"]
                message["model"] = checkpoint.name
                replacements += 1
        write_json(history / filename, session)
    metadata.update({"unique_questions": len(questions), "answer_count": replacements,
                     "session_count": len(sessions)})
    write_json(ROOT / "generation_375M.json", metadata)
    print(f"Replaced {replacements} answers in {len(sessions)} sessions. Elapsed: {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
