"""Full held-out reconstruction metrics; never ordinary language-model perplexity."""

import torch
from rapidfuzz.distance import Levenshtein
from torch.nn import functional as F

from latent_text.data import batch
from latent_text.runtime import autocast


@torch.no_grad()
def evaluate(model, rows, tokenizer, device, intervention="correct", microbatch=8):
    if not rows:
        raise ValueError("Evaluation split is empty")
    if intervention not in ("correct", "zero", "shuffle"):
        raise ValueError("Unknown memory control")
    if intervention == "shuffle" and (microbatch < 2 or len(rows) < 2):
        raise ValueError("Shuffled-memory control needs at least two examples")
    model.eval()
    nll = correct = total = exact = token_edits = char_edits = characters = 0
    vector_count = vector_bytes = utf8_bytes = 0
    failures = []
    for start in range(0, len(rows), microbatch):
        current = rows[start : start + microbatch]
        # Include a donor if the final shuffled batch contains one example.
        work = current if intervention != "shuffle" or len(current) > 1 else current + rows[:1]
        tokens, mask = batch(work, device)
        with autocast(device):
            z, lengths = model.encode(tokens, mask)
            if intervention == "zero":
                z = torch.zeros_like(z)
            elif intervention == "shuffle":
                z = z.roll(1, 0)
            logits = model.decode(z, lengths)
            target = tokens.masked_fill(~mask, -100)
            nll += F.cross_entropy(
                logits[: len(current)].float().flatten(0, 1),
                target[: len(current)].flatten(),
                reduction="sum",
            ).item()
            logits[..., :3] = -torch.inf
            ids = logits.argmax(-1)
        for i, row in enumerate(current):
            reference = row["ids"]
            predicted = ids[i, : len(reference)].tolist()
            text = tokenizer.decode(predicted, skip_special_tokens=False)
            total += len(reference)
            correct += sum(a == b for a, b in zip(reference, predicted))
            exact += text == row["text"]
            token_edits += Levenshtein.distance(reference, predicted)
            char_edits += Levenshtein.distance(row["text"], text)
            characters += len(row["text"])
            count = (len(reference) + model.config.span - 1) // model.config.span
            vector_count += count
            vector_bytes += count * model.config.width * z.element_size()
            utf8_bytes += len(row["text"].encode("utf-8"))
            if text != row["text"] and len(failures) < 10:
                failures.append({"input": row["text"], "output": text})
        # Release this batch before computing the next one; math/order unchanged.
        del tokens, mask, z, lengths, logits, target, ids
    return {
        "summary": {
            "examples": len(rows),
            "exact": exact,
            "exact_match": exact / len(rows),
            "nll": nll / total,
            "token_accuracy": correct / total,
            "token_edit_rate": token_edits / total,
            "character_edit_rate": char_edits / max(1, characters),
        },
        "payload": {
            "input_tokens": total,
            "vectors": vector_count,
            "vector_bytes": vector_bytes,
            "length_bytes": len(rows) * 8,
            "original_utf8_bytes": utf8_bytes,
            "token_id_bytes_int32": total * 4,
        },
        "intervention": intervention,
        "first_failures": failures,
        "note": "Counts exclude file container/identity bytes. Shuffling preserves target lengths.",
    }
