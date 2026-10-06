"""Text/latent interface, including encoder-only export and bounded-window input.

Chunking accepts long strings without a single long attention window. It does not
establish unlimited semantic memory or guarantee recovery outside the training distribution.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import nn

from .data import hash_text, token_ids
from .model import Config, Model
from .runtime import artifact_path, choose_device, digest, precision_for


def state_identity(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    h.update(json.dumps(asdict(model.config), sort_keys=True).encode())
    return h.hexdigest()


class EncoderOnly(nn.Module):
    """Only the trained input embedding, positions, encoder blocks and compression map."""

    def __init__(self, full_model):
        super().__init__()
        self.config = full_model.config
        for name in ("embedding", "position", "encoder", "encoder_norm", "compress"):
            setattr(self, name, getattr(full_model, name))

    def forward(self, tokens, mask):
        return Model.encode(self, tokens, mask)


class Codec:
    def __init__(self, model, tokenizer, device="cpu", precision="fp32", decoder_id=None):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.tokenizer.encode_special_tokens = True
        self.device, self.precision = device, precision
        self.decoder_id = decoder_id or state_identity(model)
        self.tokenizer_id = hash_text(tokenizer.to_str())

    @classmethod
    def load(cls, checkpoint, device="auto", precision=None):
        # Use only locally generated/trusted training checkpoints.
        device = choose_device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = Config(**state["protocol"]["model"])
        model = Model(config)
        model.load_state_dict(state["model"])
        return cls(
            model,
            Tokenizer.from_str(state["tokenizer"]),
            device,
            precision or precision_for(device),
        )

    def autocast(self):
        return torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.precision == "bf16")

    @torch.no_grad()
    def encode(self, text):
        ids = token_ids(self.tokenizer, text)
        chunks = []
        limit = self.model.config.max_tokens
        for start in range(0, len(ids), limit):
            tokens = torch.tensor([ids[start : start + limit]], device=self.device)
            with self.autocast():
                if isinstance(self.model, EncoderOnly):
                    z, lengths = self.model(tokens, torch.ones_like(tokens, dtype=torch.bool))
                else:
                    z, lengths = self.model.encode(
                        tokens, torch.ones_like(tokens, dtype=torch.bool)
                    )
            chunks.append({"vectors": z.cpu(), "lengths": lengths.cpu()})
        # No token IDs, source string, residual correction stream or target embeddings.
        return {
            "format": "position-memory-v1",
            "decoder_id": self.decoder_id,
            "tokenizer_id": self.tokenizer_id,
            "chunks": chunks,
        }

    @torch.no_grad()
    def decode(self, payload):
        if isinstance(self.model, EncoderOnly):
            raise ValueError("Encoder-only export cannot reconstruct; load its matching decoder")
        if (
            payload.get("format") != "position-memory-v1"
            or payload["decoder_id"] != self.decoder_id
            or payload["tokenizer_id"] != self.tokenizer_id
        ):
            raise ValueError("Memory format, model or tokenizer identity mismatch")
        output = []
        for chunk in payload["chunks"]:
            with self.autocast():
                ids = self.model.generate(
                    chunk["vectors"].to(self.device, dtype=self.model.embedding.weight.dtype),
                    chunk["lengths"].to(self.device),
                )
            output.extend(ids[0, : int(chunk["lengths"][0])].cpu().tolist())
        # Decode after concatenating token IDs, preserving UTF-8 sequences crossing boundaries.
        return self.tokenizer.decode(output, skip_special_tokens=False)

    def measurements(self, text, payload):
        tokens = len(token_ids(self.tokenizer, text))
        slots = sum(c["vectors"].shape[1] for c in payload["chunks"])
        features = sum(c["vectors"].numel() for c in payload["chunks"])
        vector_bytes = sum(
            c["vectors"].numel() * c["vectors"].element_size() for c in payload["chunks"]
        )
        length_bytes = sum(
            c["lengths"].numel() * c["lengths"].element_size() for c in payload["chunks"]
        )
        return {
            "input_tokens": tokens,
            "output_vectors": slots,
            "features_per_vector": self.model.config.width,
            "output_features": features,
            "tokens_per_vector": tokens / slots if slots else None,
            "vector_bytes": vector_bytes,
            "length_metadata_bytes": length_bytes,
            "identity_metadata_bytes": len(
                json.dumps({k: v for k, v in payload.items() if k != "chunks"}).encode()
            ),
            "original_utf8_bytes": len(text.encode()),
            "token_id_bytes_int32": tokens * 4,
            "chunks": len(payload["chunks"]),
            "precision": self.precision,
        }

    def reconstruct(self, text):
        payload = self.encode(text)
        output = self.decode(payload)
        return {
            "input": text,
            "output": output,
            "exact_match": output == text,
            **self.measurements(text, payload),
        }

    def save_memory(self, text, path):
        path = artifact_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encode(text), path)
        return path

    def recover_file(self, path):
        return self.decode(torch.load(path, map_location="cpu", weights_only=True))

    def export_encoder(self, path):
        path = artifact_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoder = self.model if isinstance(self.model, EncoderOnly) else EncoderOnly(self.model)
        torch.save(
            {
                "format": "position-encoder-v1",
                "config": asdict(encoder.config),
                "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "tokenizer": self.tokenizer.to_str(),
                "decoder_id": self.decoder_id,
                "architecture_sha256": digest(Path(__file__).with_name("model.py")),
            },
            path,
        )
        return path

    @classmethod
    def load_encoder(cls, path, device="cpu", precision=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format"] != "position-encoder-v1":
            raise ValueError("Unexpected encoder format")
        if state["architecture_sha256"] not in {
            digest(Path(__file__).with_name("model.py")),
        }:
            raise ValueError("Encoder architecture source changed")
        encoder = EncoderOnly(Model(Config(**state["config"])))
        encoder.load_state_dict(state["encoder"])
        return cls(
            encoder,
            Tokenizer.from_str(state["tokenizer"]),
            device,
            precision or precision_for(device),
            state["decoder_id"],
        )
