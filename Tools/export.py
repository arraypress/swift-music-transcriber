# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.2",
#     "torch==2.13.0",
#     "numpy",
#     "safetensors",
#     "huggingface_hub",
#     "einops",
#     "mido",
#     "packaging",
#     "soundfile",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Export a MuScriptor checkpoint to a Core AI .aimodel for MusicTranscriber.

    uv run Tools/export.py --size medium                 # fp32, the default
    uv run Tools/export.py --size large --dtype float16
    uv run Tools/export.py --size small --install        # copy into ~/Library/Application Support/scribe/models

Needs: a Hugging Face login that has accepted the model licence
(https://huggingface.co/MuScriptor), and a checkout of the two reference
repos beside this one or given with --muscriptor / --coreai-models:
  git clone https://github.com/muscriptor/muscriptor
  git clone https://github.com/apple/coreai-models

What it does. MuScriptor's decoder is re-authored here with the same weights
and maths but the attention cache as explicit Core AI state, then exported as
ONE asset with three functions:

  main   (inputs_embeds [1,Q,D], position_ids [1,S]; state k_cache, v_cache) -> logits [1,1,card]
  prefix (mel [1,F,512] log-mel, inst_tokens [1,L] int32)                    -> embeds [1,F+1+L,D]
  embed  (tokens [1,T] int32)                                                -> embeds [1,T,D]

The prefix order is [mel, dataset_null, instrument_group]: LMModel.forward
PREPENDS each condition in tokenize order, so self_wav ends up first. The
last mel frame is masked to zero (upstream's length mask covers 500 of 501).
Instrument tokens are class id + 1, or 0 for "no conditioning"; the graph adds
one more, as ClassConditioner.forward does.

Verified 2026-09-23 (Tools/README.md): fp32 and fp16 assets reproduce the
upstream greedy decode token for token on the demo clip for medium and large;
small fp16 diverges where upstream's own fp16 run diverges.
"""
import argparse, importlib.util, json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def _find(name: str, override: str | None) -> Path:
    for candidate in ([Path(override)] if override else []) + [HERE.parent.parent / name, HERE.parent / name, Path.home() / "Developer" / name]:
        if candidate.exists():
            return candidate
    sys.exit(f"cannot find a checkout of {name}; clone it beside this repo or pass --{name.replace('_', '-')}")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--size", default="medium", choices=["small", "medium", "large"])
ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--install", action="store_true", help="also copy the asset into the scribe models directory")
ap.add_argument("--muscriptor", help="path to a muscriptor checkout")
ap.add_argument("--coreai-models", help="path to an apple/coreai-models checkout")
args = ap.parse_args()

MUSCRIPTOR = _find("muscriptor", args.muscriptor)
COREAI_MODELS = _find("coreai-models", args.coreai_models)
sys.path.insert(0, str(MUSCRIPTOR))
sys.path.insert(0, str(COREAI_MODELS / "python" / "src"))

import coreai_torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch.composite_ops import SDPA
from coreai_models.primitives.macos.cache import KVCache
from muscriptor.modules.transformer import create_sin_embedding
from muscriptor.transcription_model import TranscriptionModel
from huggingface_hub import hf_hub_download


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# coreai_models.export/__init__ drags in transformers; load the two files needed directly.
mlir_ops = _load("mlir_ops", COREAI_MODELS / "python/src/coreai_models/export/mlir_ops.py")
externalize = _load("externalize", COREAI_MODELS / "python/src/coreai_models/export/externalize.py")

MEL_BINS = 512
N_INST_CLASSES = 1000
N_DS_CLASSES = 4


class ScribeAttention(nn.Module):
    def __init__(self, dim, heads, layer_idx):
        super().__init__()
        self.heads, self.head_dim, self.layer_idx = heads, dim // heads, layer_idx
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.sdpa = SDPA(scale=self.head_dim ** -0.5, is_causal=True)

    def forward(self, x, position_ids, cache):
        B, Q, _ = x.shape
        S = position_ids.shape[-1]
        torch._check_is_size(Q); torch._check_is_size(S)
        offset = S - Q
        torch._check_is_size(offset)
        qkv = self.in_proj(x).reshape(B, Q, 3, self.heads, self.head_dim)   # "(p h d)", p=3
        q = qkv.select(2, 0).permute(0, 2, 1, 3)
        k = qkv.select(2, 1).permute(0, 2, 1, 3)
        v = qkv.select(2, 2).permute(0, 2, 1, 3)
        k, v = cache.update_and_fetch(self.layer_idx, offset, k, v, seq_len=S, query_len=Q)
        o = self.sdpa(q, k, v).permute(0, 2, 1, 3).reshape(B, Q, self.heads * self.head_dim)
        return self.out_proj(o)


class ScribeLayer(nn.Module):
    def __init__(self, dim, heads, layer_idx):
        super().__init__()
        self.self_attn = ScribeAttention(dim, heads, layer_idx)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.linear1 = nn.Linear(dim, 4 * dim, bias=False)
        self.linear2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x, position_ids, cache):
        x = x + self.self_attn(self.norm1(x), position_ids, cache)
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))


class ScribeCore(nn.Module):
    def __init__(self, dim, heads, layers, card, max_ctx, use_pos=True, use_narrow=True):
        super().__init__()
        self.use_pos, self.use_narrow = use_pos, use_narrow
        self.layers = nn.ModuleList([ScribeLayer(dim, heads, i) for i in range(layers)])
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.head = nn.Linear(dim, card, bias=False)
        pos = torch.arange(max_ctx).view(1, -1, 1)
        self.register_buffer("pos_table", create_sin_embedding(pos, dim, 10000).squeeze(0), persistent=True)

    def forward(self, inputs_embeds, position_ids, k_cache, v_cache):
        cache = KVCache(k_cache, v_cache)
        B, Q, D = inputs_embeds.shape
        S = position_ids.shape[-1]
        torch._check_is_size(Q); torch._check_is_size(S)
        offset = S - Q
        torch._check_is_size(offset)
        pos = position_ids.narrow(-1, offset, Q).reshape(-1)
        x = inputs_embeds
        if self.use_pos:
            x = x + torch.index_select(self.pos_table, 0, pos).to(inputs_embeds.dtype).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, position_ids, cache)
        if self.use_narrow:
            x = x.narrow(1, Q - 1, 1)
        return self.head(self.out_norm(x))


class ScribePrefix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mel_proj = nn.Linear(MEL_BINS, dim)
        self.inst_embed = nn.Embedding(N_INST_CLASSES + 1, dim)
        self.ds_embed = nn.Embedding(N_DS_CLASSES + 1, dim)

    def forward(self, mel, inst_tokens):
        m = self.mel_proj(mel.to(self.mel_proj.weight.dtype))
        Fr = m.shape[1]
        mask = (torch.arange(Fr) < Fr - 1).to(m.dtype).reshape(1, Fr, 1)   # last frame masked
        m = m * mask
        i = self.inst_embed(inst_tokens + 1)
        d = self.ds_embed(torch.ones(1, 1, dtype=torch.int32))
        return torch.cat([m, d, i], dim=1)


class ScribeEmbed(nn.Module):
    def __init__(self, dim, card):
        super().__init__()
        self.emb = nn.Embedding(card + 1, dim)

    def forward(self, tokens):
        return self.emb(tokens)


def copy_weights(ref, core: ScribeCore, prefix: ScribePrefix, embed: ScribeEmbed):
    with torch.no_grad():
        for i, layer in enumerate(ref.transformer.layers):
            c = core.layers[i]
            c.self_attn.in_proj.weight.copy_(layer.self_attn.in_proj_weight)
            c.self_attn.out_proj.weight.copy_(layer.self_attn.out_proj.weight)
            for name in ("norm1", "norm2", "linear1", "linear2"):
                getattr(c, name).load_state_dict(getattr(layer, name).state_dict())
        core.out_norm.load_state_dict(ref.out_norm.state_dict())
        core.head.weight.copy_(ref.linear.weight)
        prefix.mel_proj.load_state_dict(ref.mel_proj.state_dict())
        prefix.inst_embed.load_state_dict(ref.inst_embed.state_dict())
        prefix.ds_embed.load_state_dict(ref.ds_embed.state_dict())
        embed.emb.load_state_dict(ref.emb.state_dict())


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def make_export_fn(reference_inputs, dynamic_shapes):
    def export_fn(module):
        with torch.no_grad():
            ep = torch.export.export(module, args=(), kwargs=reference_inputs, dynamic_shapes=dynamic_shapes)
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        mlir_ops.remove_functionalization(ep)
        return ep
    return export_fn


def export(core, prefix, embed, dims, out: Path, dtype, metadata=None, only=("main", "prefix", "embed")):
    dim, heads, layers, card, max_ctx = dims["dim"], dims["heads"], dims["layers"], dims["card"], dims["max_ctx"]
    head_dim = dim // heads
    Q, S = 16, 24
    k_cache = torch.zeros(layers, 1, heads, max_ctx, head_dim, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    main_inputs = {
        "inputs_embeds": torch.randn(1, Q, dim, dtype=dtype),
        "position_ids": torch.arange(S, dtype=torch.int32).unsqueeze(0),
        "k_cache": k_cache, "v_cache": v_cache,
    }
    main_shapes = {
        "inputs_embeds": {1: torch.export.Dim("q", min=1, max=max_ctx - 1)},
        "position_ids": {1: torch.export.Dim("s", min=1, max=max_ctx)},
        "k_cache": None, "v_cache": None,
    }
    prefix_inputs = {"mel": torch.randn(1, 501, MEL_BINS), "inst_tokens": torch.tensor([[1, 37]], dtype=torch.int32)}
    prefix_shapes = {"mel": {1: torch.export.Dim("f", min=1, max=501)},
                     "inst_tokens": {1: torch.export.Dim("l", min=1, max=16)}}
    embed_inputs = {"tokens": torch.tensor([[card, 10, 20]], dtype=torch.int32)}
    embed_shapes = {"tokens": {1: torch.export.Dim("t", min=1, max=max_ctx)}}

    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    if "main" in only: converter.add_pytorch_module(
        core, export_fn=make_export_fn(main_inputs, main_shapes),
        externalize_modules=externalize.EXTERNALIZE_SPECS,
        input_names=("inputs_embeds", "position_ids"), output_names=("logits",),
        state_names=("k_cache", "v_cache"), entrypoint_name="main")
    if "prefix" in only: converter.add_pytorch_module(
        prefix, export_fn=make_export_fn(prefix_inputs, prefix_shapes),
        input_names=("mel", "inst_tokens"), output_names=("embeds",), entrypoint_name="prefix")
    if "embed" in only: converter.add_pytorch_module(
        embed, export_fn=make_export_fn(embed_inputs, embed_shapes),
        input_names=("tokens",), output_names=("embeds",), entrypoint_name="embed")
    mlir_ops.register_custom_torch_lowering(converter)
    t0 = time.time()
    program = converter.to_coreai()
    program.optimize()
    if out.exists():
        import shutil; shutil.rmtree(out) if out.is_dir() else out.unlink()
    program.save_asset(out, metadata)
    print(flush=True); print(f"[export] {out} in {time.time() - t0:.1f}s")




# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    weights = hf_hub_download(f"MuScriptor/muscriptor-{args.size}", "model.safetensors")
    hf_hub_download(f"MuScriptor/muscriptor-{args.size}", "config.json")
    tm = TranscriptionModel.load_model(weights_path=weights, device="cpu", dtype="float32")
    lm = tm._model
    dims = dict(dim=lm.dim, heads=lm.transformer.layers[0].self_attn.num_heads,
                layers=len(lm.transformer.layers), card=lm.card, max_ctx=503 + 2000)
    print(f"[export] {args.size}: {dims}", flush=True)

    dtype = getattr(torch, args.dtype)
    core = ScribeCore(dims["dim"], dims["heads"], dims["layers"], dims["card"], dims["max_ctx"]).eval()
    prefix, embed = ScribePrefix(dims["dim"]).eval(), ScribeEmbed(dims["dim"], dims["card"]).eval()
    mel_cond = lm.condition_provider.conditioners["self_wav"]
    with torch.no_grad():
        for i, layer in enumerate(lm.transformer.layers):
            c = core.layers[i]
            c.self_attn.in_proj.weight.copy_(layer.self_attn.in_proj_weight)
            c.self_attn.out_proj.weight.copy_(layer.self_attn.out_proj.weight)
            for name in ("norm1", "norm2", "linear1", "linear2"):
                getattr(c, name).load_state_dict(getattr(layer, name).state_dict())
        core.out_norm.load_state_dict(lm.out_norm.state_dict())
        core.head.weight.copy_(lm.linear.weight)
        prefix.mel_proj.load_state_dict(mel_cond.output_proj.state_dict())
        prefix.inst_embed.load_state_dict(lm.condition_provider.conditioners["instrument_group"].embed.state_dict())
        prefix.ds_embed.load_state_dict(lm.condition_provider.conditioners["dataset_name"].embed.state_dict())
        embed.emb.load_state_dict(lm.emb.state_dict())
    if dtype != torch.float32:
        core.to(dtype); prefix.to(dtype); embed.to(dtype)
        core.pos_table = core.pos_table.float()

    metadata = AIModelAssetMetadata()
    metadata.author = "Kyutai and Mirelo (MuScriptor); Core AI export by MusicTranscriber"
    metadata.license = "CC-BY-NC-4.0 (weights); see https://huggingface.co/MuScriptor"
    metadata.model_description = (f"MuScriptor {args.size} multi-instrument music transcription decoder, {args.dtype}. "
                                  "Functions: main (stateful decoder), prefix (conditioning), embed (tokens).")
    metadata.creation_date = int(time.time())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out / f"scribe-{args.size}-{args.dtype}.aimodel"
    export(core, prefix, embed, dims, asset, dtype, metadata)
    print(f"[export] wrote {asset} ({sum(f.stat().st_size for f in asset.rglob('*') if f.is_file()) / 1e6:.0f} MB)")

    if args.install:
        dest_dir = Path.home() / "Library/Application Support/scribe/models"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / asset.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(asset, dest)
        print(f"[export] installed -> {dest}")
