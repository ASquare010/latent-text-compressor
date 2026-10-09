"""Frozen split-disjoint packs and shared, bounded sampling without replacement."""

import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from tokenizers import Tokenizer

from .data import hash_text, load_data, prepare, token_ids
from .durable import OwnedLock, atomic_json, identity, read_json, sha256


def seeded(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


def length_group(n, boundaries):
    return next((b for b in boundaries if n <= b), boundaries[-1])


def sequence_id(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<u2").tobytes()).hexdigest()


def packed_rows(rows, context, separator, seed):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    buffer, provenance = [], []
    for position, i in enumerate(order):
        row = rows[i]
        contents = (separator if position else []) + row["ids"]
        while contents:
            take = min(context - len(buffer), len(contents))
            buffer.extend(contents[:take])
            contents = contents[take:]
            provenance.append(row["paragraph_id"])
            if len(buffer) == context:
                yield {
                    "ids": buffer,
                    "source": "natural_packed",
                    "parents": sorted(set(provenance)),
                }
                buffer, provenance = [], []
    if buffer:
        yield {"ids": buffer, "source": "natural_packed", "parents": sorted(set(provenance))}


def synthesize(count, context, vocab, seed, patterns, forbidden):
    rng = random.Random(seed)
    seen = set(forbidden)
    out = []
    while len(out) < count:
        n = rng.randint(32, context)
        if patterns:
            alphabet = rng.sample(range(3, vocab), rng.randint(2, 64))
            motif = rng.choices(alphabet, k=rng.randint(2, 32))
            ids = (
                (motif * math.ceil(n / len(motif)))[:n]
                if rng.random() < 0.5
                else rng.choices(alphabet, k=n)
            )
        else:
            ids = [rng.randrange(3, vocab) for _ in range(n)]
        key = sequence_id(ids)
        if key not in seen:
            out.append({"ids": ids, "source": "patterns" if patterns else "random", "parents": []})
            seen.add(key)
    return out


def difficulty_weights(rows, counts, config):
    probability = (counts + 1) / (counts.sum() + len(counts))
    surprisal = -np.log(probability)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        row["difficulty"] = float(np.mean(surprisal[row["ids"]]))
        key = (row["source"], length_group(len(row["ids"]), config["length_boundaries"]))
        groups[key].append(i)
    for members in groups.values():
        order = sorted(members, key=lambda i: rows[i]["difficulty"])
        start = 0
        while start < len(order):
            end = start + 1
            while (
                end < len(order)
                and rows[order[end]]["difficulty"] == rows[order[start]]["difficulty"]
            ):
                end += 1
            rank = (start + end - 1) / (2 * max(1, len(order) - 1)) if len(order) > 1 else 0.5
            base = config["difficulty_weight_min"] + rank * (
                config["difficulty_weight_max"] - config["difficulty_weight_min"]
            )
            rate = config["uniform_component"] + (1 - config["uniform_component"]) * base
            for index in order[start:end]:
                rows[index]["weight"] = rate
            start = end


def write_rows(folder, name, rows):
    folder = Path(folder)
    data, metadata, offset = [], [], 0
    for row in rows:
        ids = row["ids"]
        data.extend(ids)
        metadata.append(
            {k: v for k, v in row.items() if k != "ids"}
            | {"offset": offset, "length": len(ids), "id": sequence_id(ids)}
        )
        offset += len(ids)
    with (folder / f"{name}.npy").open("wb") as f:
        np.save(f, np.asarray(data, dtype="<u2"), allow_pickle=False)
        f.flush()
        import os

        os.fsync(f.fileno())
    atomic_json(folder / f"{name}.json", metadata)
    return {
        "rows": len(rows),
        "tokens": offset,
        "sources": dict(Counter(r["source"] for r in rows)),
        "length_quantiles": np.quantile(
            [len(r["ids"]) for r in rows], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
    }


class RowStore:
    def __init__(self, folder, name):
        self.meta = read_json(Path(folder) / f"{name}.json")
        self.tokens = np.load(Path(folder) / f"{name}.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta[i]
        return {
            **row,
            "ids": self.tokens[row["offset"] : row["offset"] + row["length"]].astype(np.int64),
        }


def prepare_comparison(config):
    folder = Path(config["data_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    with OwnedLock(folder / "preparation.lock"):
        manifest_path = folder / "manifest.json"
        data_recipe = {
            k: config[k] for k in ("dataset", "sampling", "packing_seed", "synthetic_seed")
        }
        if manifest_path.exists():
            manifest = verify_data(folder)
            if manifest["recipe"] != data_recipe:
                raise ValueError("Frozen data settings differ; use a new data_dir")
            return manifest
        base = folder / "base"
        if not (base / "manifest.json").exists():
            if base.exists() and any(base.iterdir()):
                raise ValueError(
                    "Incomplete data preparation retained; inspect before retrying in a new directory"
                )
            prepare(base, **config["dataset"])
        tokenizer, splits, original_manifest = load_data(base)
        if not original_manifest["complete"]:
            raise ValueError("Base dataset incomplete; retained for inspection")
        # Audit and remove whitespace/case-normalized duplicates before any training.
        # Exact deduplication and document splitting were already done by prepare().
        seen, removed, clean = set(), Counter(), {}
        for split in ("test", "valid", "train"):
            clean[split] = []
            for row in splits[split]:
                key = hash_text(" ".join(row["text"].casefold().split()))
                if key in seen:
                    removed[split] += 1
                    continue
                seen.add(key)
                clean[split].append(row)
        # Refit tokenizer on the final training-only natural corpus, so removed
        # cross-split normalized duplicates cannot enter tokenizer fitting either.
        from .data import tokenizer_for

        tokenizer = tokenizer_for(
            (r["text"] for r in clean["train"]), config["model"]["vocab_size"]
        )
        retokenized_exclusions = Counter()
        for split in clean:
            retained = []
            for row in clean[split]:
                ids = token_ids(tokenizer, row["text"])
                if config["dataset"]["min_tokens"] <= len(ids) <= config["dataset"]["max_tokens"]:
                    retained.append({**row, "ids": ids})
                else:
                    retokenized_exclusions[split] += 1
            clean[split] = retained
        tokenizer.save(str(folder / "tokenizer.json"))
        documents = {s: {r["document_id"] for r in rows} for s, rows in clean.items()}
        paragraphs = {s: {r["paragraph_id"] for r in rows} for s, rows in clean.items()}
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            if documents[a] & documents[b] or paragraphs[a] & paragraphs[b]:
                raise ValueError("Split overlap")
        freq = np.zeros(config["model"]["vocab_size"], dtype=np.int64)
        for row in clean["train"]:
            np.add.at(freq, row["ids"], 1)
        atomic_json(folder / "training-token-counts.json", freq.tolist())
        separator = token_ids(tokenizer, "\n\n")
        audit = {
            "existing_dataset": "No prepared dataset present at initial inspection; new pinned corpus",
            "base": original_manifest,
            "normalized_duplicates_removed": dict(removed),
            "retokenized_length_exclusions": dict(retokenized_exclusions),
            "counts": {s: len(r) for s, r in clean.items()},
            "document_overlap": 0,
            "paragraph_overlap": 0,
            "near_duplicates": "Exact and normalized dedup only; semantic/paraphrase duplicates unmeasured",
            "source_domains": {
                s: dict(
                    Counter(
                        urlparse(r.get("source_url") or "").netloc or "unknown" for r in rows
                    ).most_common(30)
                )
                for s, rows in clean.items()
            },
            "length_quantiles": {
                s: np.quantile(
                    [len(r["ids"]) for r in rows], [0, 0.25, 0.5, 0.75, 0.95, 1]
                ).tolist()
                for s, rows in clean.items()
            },
        }
        summaries, all_heldout = {}, set()
        # Split membership files make the exact retained base population independently auditable.
        for split, rows in clean.items():
            atomic_json(
                folder / f"{split}-membership.json",
                [
                    {
                        "paragraph_id": r["paragraph_id"],
                        "document_id": r["document_id"],
                        "tokens": len(r["ids"]),
                        "sequence_id": sequence_id(r["ids"]),
                    }
                    for r in rows
                ],
            )
        for split in ("valid", "test"):
            short = [r for r in clean[split] if len(r["ids"]) <= 256]
            random.Random(seeded(config["packing_seed"], split + "short")).shuffle(short)
            n = config["sampling"]["shared_short_eval_count"]
            if len(short) < n:
                raise ValueError(f"Need {n} shared short {split} rows; found {len(short)}")
            short = [
                {"ids": r["ids"], "source": "natural_short", "parents": [r["paragraph_id"]]}
                for r in short[:n]
            ]
            summaries[f"{split}-short"] = write_rows(folder, f"{split}-short", short)
            all_heldout.update(sequence_id(r["ids"]) for r in clean[split])
            for context in (512, 1024):
                packs = list(
                    packed_rows(
                        clean[split], context, separator, seeded(config["packing_seed"], split)
                    )
                )
                packs = packs[: config["sampling"]["packed_eval_count"]]
                summaries[f"{split}-{context}"] = write_rows(folder, f"{split}-{context}", packs)
                all_heldout.update(sequence_id(r["ids"]) for r in packs)
        for context in (512, 1024):
            short = [
                {"ids": r["ids"], "source": "natural_short", "parents": [r["paragraph_id"]]}
                for r in clean["train"]
                if len(r["ids"]) <= 256
            ]
            packs = list(
                packed_rows(
                    clean["train"], context, separator, seeded(config["packing_seed"], "train")
                )
            )
            rows = short + packs
            if any(sequence_id(r["ids"]) in all_heldout for r in rows):
                raise ValueError("Packed training/held-out token collision")
            forbidden = all_heldout | {sequence_id(r["ids"]) for r in rows}
            for patterns in (True, False):
                added = synthesize(
                    config["sampling"]["synthetic_rows_per_source"],
                    context,
                    tokenizer.get_vocab_size(),
                    seeded(config["synthetic_seed"], f"{context}:{patterns}"),
                    patterns,
                    forbidden,
                )
                rows.extend(added)
                forbidden.update(sequence_id(r["ids"]) for r in added)
            difficulty_weights(rows, freq, config["sampling"])
            summaries[f"train-{context}"] = write_rows(folder, f"train-{context}", rows)
            del rows, short, packs
        atomic_json(folder / "audit.json", audit)
        manifest = {
            "format": "comparison-data-v1",
            "recipe": data_recipe,
            "audit": audit,
            "summaries": summaries,
            "files": {
                p.name: sha256(p)
                for p in sorted(folder.iterdir())
                if p.is_file()
                and p.suffix in (".npy", ".json")
                and not p.name.endswith("owner.json")
            },
        }
        atomic_json(manifest_path, manifest)
        verify_data(folder)
        return manifest


def verify_data(folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    for name, checksum in manifest["files"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Frozen dataset changed: {name}")
    return manifest


class SharedSampler:
    """Model-independent permutations and quotas; every source exhausts before reuse."""

    def __init__(self, metadata, policy, seed):
        self.meta, self.policy = metadata, policy
        self.rng = random.Random(seed)
        self.source_rng = {s: random.Random(seeded(seed, s)) for s in policy["cycle_counts"]}
        self.members = {
            s: [i for i, r in enumerate(metadata) if r["source"] == s] for s in self.source_rng
        }
        if any(not items for items in self.members.values()):
            raise ValueError("Every sampling source must be nonempty")
        self.orders, self.cursor, self.epochs = {}, {}, {}
        self.cycle, self.cycle_cursor = [], 0

    def take(self, count):
        result = []
        for _ in range(count):
            if self.cycle_cursor == len(self.cycle):
                self.cycle = [s for s, n in self.policy["cycle_counts"].items() for _ in range(n)]
                self.rng.shuffle(self.cycle)
                self.cycle_cursor = 0
            source = self.cycle[self.cycle_cursor]
            self.cycle_cursor += 1
            if self.cursor.get(source, 0) == len(self.orders.get(source, [])):
                rng = self.source_rng[source]
                self.orders[source] = sorted(
                    self.members[source],
                    key=lambda i: (
                        -math.log(max(rng.random(), 1e-300)) / self.meta[i]["weight"],
                        i,
                    ),
                )
                self.cursor[source] = 0
                self.epochs[source] = self.epochs.get(source, 0) + 1
            result.append(self.orders[source][self.cursor[source]])
            self.cursor[source] += 1
        return result

    def state_dict(self):
        return {
            "rng": self.rng.getstate(),
            "source_rng": {k: v.getstate() for k, v in self.source_rng.items()},
            "orders": self.orders,
            "cursor": self.cursor,
            "epochs": self.epochs,
            "cycle": self.cycle,
            "cycle_cursor": self.cycle_cursor,
            "metadata_hash": identity(self.meta),
        }

    def load_state_dict(self, state):
        if state["metadata_hash"] != identity(self.meta):
            raise ValueError("Sampler data changed")
        self.rng.setstate(state["rng"])
        for key, value in state["source_rng"].items():
            self.source_rng[key].setstate(value)
        for key in ("orders", "cursor", "epochs", "cycle", "cycle_cursor"):
            setattr(self, key, state[key])


def load_tokenizer(folder):
    return Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
