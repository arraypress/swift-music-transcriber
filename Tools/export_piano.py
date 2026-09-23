# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "coreai-torch==0.4.2", "coreai-core==1.0.0b2", "piano_transcription_inference", "torchlibrosa", "librosa", "matplotlib", "mido", "audioread", "soundfile"]
# ///
"""ByteDance piano transcription (Kong et al. 2020, Apache 2.0) → scribe-piano-float32.aimodel.

    uv run Tools/export_piano.py [--checkpoint path.pth] [--out Tools/exports] [--install]

Downloads the checkpoint `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth` (172 MB) from Zenodo when
not given. The asset has two entry points:

  trunk       log-mel [1, 1001, 229] (one 10 s segment at 100 fps) → the seven CRNN trunks'
              features [1, 1001, 768] each, in the order frame, reg_onset, reg_offset, velocity,
              reg_pedal_onset, reg_pedal_offset, reg_pedal_frame: bn0, four conv blocks, fc5,
              bn5, relu — exactly upstream's forward up to the GRU input.
  parameters  every GRU and head weight as one flat float vector, in a fixed order (per head: per
              layer, forward then reverse, weight_ih, weight_hh, bias_ih, bias_hh; then the linear
              layer's weight and bias; heads in trunk order, then reg_onset_gru/fc, frame_gru/fc).

Core AI has no recurrent op; a decomposed GRU unrolls to ~770k ops at this size. The sixteen
bidirectional recurrences therefore run in Swift (`GRU.swift`, Accelerate) from `parameters`.
Nothing is re-authored: the trunk is asserted equal to upstream's activations before export, and
the Swift tests hold the whole pipeline to upstream's events on real audio.
"""
import argparse, os, shutil, sys, time, urllib.request
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coreai_torch
from coreai.runtime import AIModelAssetMetadata
from piano_transcription_inference.models import Note_pedal

HERE = Path(__file__).resolve().parent
ZENODO = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
BRANCHES = ["frame_model", "reg_onset_model", "reg_offset_model", "velocity_model",
            "reg_pedal_onset_model", "reg_pedal_offset_model", "reg_pedal_frame_model"]
T = 1001

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=str(HERE / "exports" / "piano_checkpoint.pth"))
ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--install", action="store_true", help="copy the asset into ~/Library/Application Support/scribe/models")
args = ap.parse_args()

ckpt = Path(args.checkpoint)
if not ckpt.exists():
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] downloading the checkpoint (172 MB) to {ckpt}", flush=True)
    urllib.request.urlretrieve(ZENODO, ckpt)

model = Note_pedal(frames_per_second=100, classes_num=88)
model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=False)
model.eval()
note, pedal = model.note_model, model.pedal_model
def branch(name): return getattr(note, name) if hasattr(note, name) else getattr(pedal, name)

class Trunk(nn.Module):
    """bn0 + the seven convolutional trunks, upstream's forward up to the GRU input."""
    def __init__(self):
        super().__init__()
        self.bn0_note, self.bn0_pedal = note.bn0, pedal.bn0
        self.branches = nn.ModuleList([branch(b) for b in BRANCHES])
    def forward(self, mel):                       # [1, T, 229]
        x = mel.unsqueeze(1)                      # [1, 1, T, 229]
        outs = []
        for i, b in enumerate(self.branches):
            bn = self.bn0_note if i < 4 else self.bn0_pedal
            y = bn(x.transpose(1, 3)).transpose(1, 3)
            y = b.conv_block1(y, pool_size=(1, 2)); y = b.conv_block2(y, pool_size=(1, 2))
            y = b.conv_block3(y, pool_size=(1, 2)); y = b.conv_block4(y, pool_size=(1, 2))
            y = y.transpose(1, 2).flatten(2)
            y = F.relu(b.bn5(b.fc5(y).transpose(1, 2)).transpose(1, 2))
            outs.append(y)
        return tuple(outs)

def gru_params(gru):
    out = []
    for layer in range(gru.num_layers):
        for suffix in ["", "_reverse"]:
            for p in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
                out.append(getattr(gru, f"{p}_l{layer}{suffix}").detach().flatten())
    return out
flat = []
for name in BRANCHES:
    b = branch(name)
    flat += gru_params(b.gru) + [b.fc.weight.detach().flatten(), b.fc.bias.detach().flatten()]
for gru, fc in [(note.reg_onset_gru, note.reg_onset_fc), (note.frame_gru, note.frame_fc)]:
    flat += gru_params(gru) + [fc.weight.detach().flatten(), fc.bias.detach().flatten()]
parameters = torch.cat(flat)

class Parameters(nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer("p", parameters.clone())
    def forward(self, probe):                     # a one-element input keeps torch.export happy
        return self.p + probe * 0

trunk = Trunk().eval()
# The trunk must reproduce upstream's own forward exactly on real input before anything is exported.
mel = torch.randn(1, T, 229) * 20 - 40
with torch.no_grad():
    ours = trunk(mel)
    x = mel.unsqueeze(1)
    for i, name in enumerate(BRANCHES):
        m = note if i < 4 else pedal
        y = m.bn0(x.transpose(1, 3)).transpose(1, 3)
        b = branch(name)
        y = b.conv_block1(y, pool_size=(1, 2)); y = b.conv_block2(y, pool_size=(1, 2)); y = b.conv_block3(y, pool_size=(1, 2)); y = b.conv_block4(y, pool_size=(1, 2))
        y = y.transpose(1, 2).flatten(2); y = F.relu(b.bn5(b.fc5(y).transpose(1, 2)).transpose(1, 2))
        assert torch.equal(y, ours[i]), name
print(f"[export] trunk equals upstream's forward on all seven branches; parameters {parameters.numel()/1e6:.2f} M floats", flush=True)

t0 = time.time()
with torch.no_grad():
    ex_trunk = torch.export.export(trunk, (mel,)).run_decompositions(coreai_torch.get_decomp_table())
    ex_params = torch.export.export(Parameters().eval(), (torch.zeros(1),)).run_decompositions(coreai_torch.get_decomp_table())
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
converter.add_exported_program(ex_trunk, input_names=["mel"], output_names=BRANCHES, entrypoint_name="trunk")
converter.add_exported_program(ex_params, input_names=["probe"], output_names=["parameters"], entrypoint_name="parameters")
program = converter.to_coreai(); program.optimize()
meta = AIModelAssetMetadata()
meta.author = "Kong, Li, Song, Wang (ByteDance) — piano_transcription; Core AI export by MusicTranscriber"
meta.license = "Apache-2.0"
meta.model_description = "High-resolution piano transcription with pedals (Kong et al. 2020), checkpoint note_F1=0.9677_pedal_F1=0.9186: trunk = log-mel [1,1001,229] → seven CRNN trunks [1,1001,768]; parameters = GRU and head weights, flat; recurrences run in the host."
meta.creation_date = int(time.time())
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
asset = out / "scribe-piano-float32.aimodel"
if asset.exists(): shutil.rmtree(asset)
program.save_asset(asset, meta)
print(f"[export] saved {asset} ({sum(f.stat().st_size for f in asset.rglob('*'))/1e6:.0f} MB) in {time.time()-t0:.0f} s", flush=True)
if args.install:
    dest = Path.home() / "Library/Application Support/scribe/models" / asset.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(asset, dest); print(f"[export] installed {dest}")
