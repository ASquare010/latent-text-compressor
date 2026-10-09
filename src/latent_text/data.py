"""Reversible text and bounded, document-disjoint streaming preparation."""

import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .runtime import artifact_path, digest, write_json


def tokenizer_for(texts, vocab_size=4096):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<bos>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return tokenizer


def token_ids(tokenizer, text):
    tokenizer.encode_special_tokens = True  # Literal '<eos>' in text is not a control code.
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if tokenizer.decode(ids, skip_special_tokens=False) != text:
        raise ValueError("Tokenizer did not preserve the exact input")
    return ids


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(
    output,
    documents=None,
    *,
    repo="HuggingFaceFW/fineweb-edu",
    subset="sample-10BT",
    counts=None,
    min_tokens=32,
    max_tokens=256,
    vocab_size=4096,
    max_documents=500_000,
    revision=None,
):
    """Use a list of documents, a UTF-8 text file, or None to stream FineWeb-Edu."""
    counts = counts or {"train": 50_000, "valid": 1000, "test": 1000}
    if set(counts) != {"train", "valid", "test"} or any(
        type(n) is not int or n < 1 for n in counts.values()
    ):
        raise ValueError("Provide positive train, valid and test paragraph counts")
    if not 1 <= min_tokens <= max_tokens or vocab_size < 259 or max_documents < 1:
        raise ValueError("Invalid token limits, document budget or byte-BPE vocabulary size")
    output = artifact_path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Data directory is not empty; load it or choose a new version")
    output.mkdir(parents=True, exist_ok=True)
    counts = counts or {"train": 50_000, "valid": 1000, "test": 1000}
    remote = documents is None
    if remote:
        from datasets import load_dataset

        if not revision:
            raise ValueError("Supply a pinned dataset revision in the config")
        documents = load_dataset(repo, subset, revision=revision, split="train", streaming=True)
    elif isinstance(documents, (str, Path)):
        with Path(documents).open(encoding="utf-8", newline="") as stream:
            documents = re.split(r"\r?\n\r?\n", stream.read())
        revision = None
    if not remote:
        revision = None
    source = {
        "repo": repo if revision else "local",
        "subset": subset if revision else None,
        "revision": revision,
    }
    write_json(output / "source.json", source)
    rows = {k: [] for k in counts}
    seen_documents, seen_paragraphs = set(), set()
    scanned = skipped = 0
    for item in documents:
        if scanned >= max_documents:
            break
        text = item if isinstance(item, str) else item["text"]
        doc_id = hash_text(text)
        scanned += 1
        if doc_id in seen_documents:
            continue
        seen_documents.add(doc_id)
        bucket = int(doc_id[:8], 16) % 100
        split = "test" if bucket == 0 else "valid" if bucket == 1 else "train"
        if len(rows[split]) < counts[split] * 4:
            for match in re.finditer(r"[^\n]+", text):
                paragraph = match.group()
                key = hash_text(paragraph)
                if not paragraph.strip() or key in seen_paragraphs:
                    skipped += 1
                    continue
                seen_paragraphs.add(key)
                if 40 <= len(paragraph) <= max_tokens * 12:
                    rows[split].append(
                        {
                            "text": paragraph,
                            "document_id": doc_id,
                            "paragraph_id": key,
                            "source_id": None if isinstance(item, str) else item.get("id"),
                            "source_url": None if isinstance(item, str) else item.get("url"),
                            "offset": match.start(),
                        }
                    )
                if len(rows[split]) >= counts[split] * 4:
                    break
        if scanned % 5000 == 0:
            print("Preparing", scanned, {k: len(v) for k, v in rows.items()}, flush=True)
        if all(len(rows[k]) >= counts[k] * 4 for k in counts) or scanned >= max_documents:
            break
    if not rows["train"]:
        raise ValueError("No training paragraphs; provide more documents")
    tokenizer = tokenizer_for((r["text"] for r in rows["train"]), vocab_size)
    tokenizer.save(str(output / "tokenizer.json"))
    excluded, selected = {}, {}
    for split, candidates in rows.items():
        kept = []
        excluded[split] = {"too_short": 0, "too_long": 0}
        for row in candidates:
            ids = token_ids(tokenizer, row["text"])
            if not min_tokens <= len(ids) <= max_tokens:
                excluded[split]["too_short" if len(ids) < min_tokens else "too_long"] += 1
                continue
            if len(kept) < counts[split]:
                kept.append({**row, "ids": ids})
        selected[split] = len(kept)
        with (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in kept:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        **source,
        "requested_counts": counts,
        "counts": selected,
        "complete": selected == counts,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "scanned_documents": scanned,
        "duplicates_or_empty": skipped,
        "excluded": excluded,
        "tokenizer_sha256": digest(output / "tokenizer.json"),
        "split_sha256": {s: digest(output / f"{s}.jsonl") for s in counts},
    }
    write_json(output / "manifest.json", manifest)
    if not manifest["complete"]:
        print("Fewer eligible paragraphs than requested:", selected, flush=True)
    return output


def load_data(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    for split, expected in manifest["split_sha256"].items():
        if digest(path / f"{split}.jsonl") != expected:
            raise ValueError("Dataset content changed")
    if digest(path / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    rows = {
        s: [
            json.loads(line)
            for line in (path / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for s in manifest["counts"]
    }
    return tokenizer, rows, manifest


def batch(rows, device):
    lengths = [len(row["ids"]) for row in rows]
    tokens = torch.zeros(len(rows), max(lengths), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        ids = torch.tensor(row["ids"], device=device)
        tokens[i, : len(ids)] = ids
    return tokens, tokens.ne(0)
