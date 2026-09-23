# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.2",
#     "torch==2.13.0",
#     "beat-this>=1.1",
#     "numpy",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Export the Beat This! beat tracker (CPJKU, MIT) to a Core AI .aimodel.

    uv run Tools/export_beat_this.py --install

MuScriptor detects tempo and metre with Beat This!'s `final0` checkpoint; with
this asset installed, scribe uses the same tracker and the MIDI grid matches
upstream's. The original module exports unchanged: `spect [1,T,128]` log-mel
in (22.05 kHz, 50 frames/s), framewise `beat` and `downbeat` logits `[1,T]`
out, T dynamic up to the 1,500-frame chunks upstream feeds it.
"""
import argparse, shutil, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
import coreai_torch
from coreai.runtime import AIModelAssetMetadata
from beat_this.inference import load_model
from beat_this.model import roformer


def rope(t, freqs):
    """rotary_embedding_torch's rotate_queries_or_keys: interleaved pairs, positions 0..n-1,
    the half-rotation as a constant matrix instead of a split-and-stack."""
    n, d = t.shape[-2], t.shape[-1]
    pos = torch.arange(n, dtype=torch.float32)
    ang = (pos[:, None] * freqs[None, :]).repeat_interleave(2, dim=-1)
    P = torch.zeros(d, d)
    for i in range(0, d, 2):
        P[i + 1, i] = -1.0
        P[i, i + 1] = 1.0
    return t * ang.cos() + (t @ P) * ang.sin()


def attention_forward(self, x):
    x = self.norm(x)
    b, n, _ = x.shape
    h, dh = self.heads, self.to_qkv.weight.shape[0] // (3 * self.heads)
    qkv = self.to_qkv(x).reshape(b, n, 3, h, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if self.rotary_embed is not None:
        q, k = rope(q, self.rotary_embed.freqs), rope(k, self.rotary_embed.freqs)
    out = F.scaled_dot_product_attention(q, k, v)
    if self.to_gates is not None:
        out = out * torch.sigmoid(self.to_gates(x)).permute(0, 2, 1).reshape(b, h, n, 1)
    return self.to_out(out.permute(0, 2, 1, 3).reshape(b, n, h * dh))


_original_attention_forward = roformer.Attention.forward

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", default="final0", help="Beat This! checkpoint name, path or URL")
ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--install", action="store_true", help="also copy the asset into the scribe models directory")
args = ap.parse_args()


class Wrapped(torch.nn.Module):
    """The model's dict output as two named tensors."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, spect):
        out = self.m(spect)
        return out["beat"], out["downbeat"]


original = load_model(args.checkpoint).eval()
model = Wrapped(original).eval()
# The re-authored attention must equal the original's on real-sized input.
probe = torch.randn(1, 600, 128)
with torch.no_grad():
    reference = model(probe)[0]
    roformer.Attention.forward = attention_forward
    patched = model(probe)[0]
worst = float((reference - patched).abs().max())
print(f"[export] Beat This! {args.checkpoint}: {sum(p.numel() for p in original.parameters()) / 1e6:.1f}M parameters; re-authored attention max |diff| {worst:.1e}")
assert worst < 1e-3, "the re-authored attention does not match the original"
example = torch.randn(1, 1500, 128)
dynamic = {"spect": {1: torch.export.Dim("t", min=8, max=1500)}}
t0 = time.time()
with torch.no_grad():
    exported = torch.export.export(model, args=(), kwargs={"spect": example}, dynamic_shapes=dynamic)
exported = exported.run_decompositions(coreai_torch.get_decomp_table())
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
converter.add_exported_program(exported, input_names=["spect"], output_names=["beat", "downbeat"], entrypoint_name="main")
program = converter.to_coreai()
program.optimize()

metadata = AIModelAssetMetadata()
metadata.author = "Foscarin, Schlüter, Widmer (CPJKU) — Beat This!; Core AI export by MusicTranscriber"
metadata.license = "MIT"
metadata.model_description = f"Beat This! beat and downbeat tracker, checkpoint {args.checkpoint}: log-mel [1,T,128] at 50 fps in, framewise beat/downbeat logits out."
metadata.creation_date = int(time.time())

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
asset = out / "scribe-beat-this-float32.aimodel"
if asset.exists():
    shutil.rmtree(asset)
program.save_asset(asset, metadata)
print(f"[export] wrote {asset} in {time.time() - t0:.1f}s")
if args.install:
    dest_dir = Path.home() / "Library/Application Support/scribe/models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / asset.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(asset, dest)
    print(f"[export] installed -> {dest}")
