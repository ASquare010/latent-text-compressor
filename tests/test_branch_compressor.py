"""Branch compressor behavior and checkpoint compatibility regression checks."""

import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from latent_text.codec import Codec
from latent_text.ffn import CompressorBranchFFN
from latent_text.model import Config, Model


class BranchCompressorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = Config(
            vocab_size=32,
            width=16,
            heads=2,
            encoder_layers=1,
            decoder_layers=1,
            max_tokens=16,
            span=4,
        )

    def test_padding_gradients_and_saved_calibration(self):
        model = Model(self.config, seed=913)
        self.assertTrue(
            all(isinstance(b.ffn, CompressorBranchFFN) for b in [*model.encoder, *model.decoder])
        )
        x = torch.tensor([[3, 4, 5, 6, 7]])
        mask = torch.ones_like(x, dtype=torch.bool)
        z, n = model.encode(x, mask)
        xp = torch.nn.functional.pad(x, (0, 3), value=19)
        zp, np = model.encode(xp, torch.arange(8)[None] < 5)
        torch.testing.assert_close(z, zp)
        torch.testing.assert_close(n, np)
        y = model(x, mask)
        y.square().mean().backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )
        restored = Model(self.config, seed=27)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(x, mask), y, rtol=0, atol=0)

    def test_codec_roundtrip_and_legacy_rejection(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path("artifacts").mkdir()
                tokenizer = Tokenizer(
                    WordLevel(
                        {"<pad>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "[UNK]": 4},
                        unk_token="[UNK]",
                    )
                )
                model = Model(self.config, seed=913)
                codec = Codec(model, tokenizer)
                checkpoint = Path("artifacts/model.pt")
                state = {
                    "protocol": {"model": asdict(self.config)},
                    "model": model.state_dict(),
                    "tokenizer": tokenizer.to_str(),
                }
                torch.save(state, checkpoint)
                loaded = Codec.load(checkpoint, device="cpu")
                memory = codec.encode("hello")
                torch.testing.assert_close(
                    loaded.encode("hello")["chunks"][0]["vectors"],
                    memory["chunks"][0]["vectors"],
                    rtol=0,
                    atol=0,
                )
                path = codec.export_encoder("artifacts/encoder.pt")
                encoder = Codec.load_encoder(path, device="cpu")
                self.assertEqual(loaded.decode(encoder.encode("hello")), codec.decode(memory))
                exported = torch.load(path, weights_only=True)
                exported["architecture_sources"]["1"] = "changed"
                torch.save(exported, path)
                with self.assertRaisesRegex(ValueError, "source changed"):
                    Codec.load_encoder(path, device="cpu")
                del state["protocol"]["model"]["ffn"]
                torch.save(state, checkpoint)
                with self.assertRaisesRegex(ValueError, "Legacy CurveFFN"):
                    Codec.load(checkpoint, device="cpu")
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
