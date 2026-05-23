from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from spdnet.models import SPD
from spdnet.utils.config import load_config, merge_overrides
from spdnet.utils.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Report SPD parameters, FLOPs, latency, FPS, and memory.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load before benchmarking")
    parser.add_argument("--cfg-options", nargs="*", default=None)
    parser.add_argument("--input-size", type=int, nargs=4, default=[1, 3, 512, 512], metavar=("N", "C", "H", "W"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--deploy", action="store_true", help="Fuse re-parameterized encoder branches before benchmarking")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def normalize_state_dict(state_dict):
    if state_dict and all(k.startswith("decode_head.") for k in state_dict.keys()):
        return {k.removeprefix("decode_head."): v for k, v in state_dict.items()}
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.image_encoder.parameters()) if hasattr(model, "image_encoder") else 0
    decoder = sum(p.numel() for p in model.mask_decoder.parameters()) if hasattr(model, "mask_decoder") else 0
    return {
        "params": int(total),
        "trainable_params": int(trainable),
        "encoder_params": int(encoder),
        "decoder_params": int(decoder),
        "params_m": total / 1e6,
    }


def hook_flops(model: nn.Module, inputs: torch.Tensor) -> int:
    flops = 0
    handles = []

    def conv_hook(module, inp, out):
        nonlocal flops
        x = inp[0]
        batch = x.shape[0]
        out_h, out_w = out.shape[-2:]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        flops += int(batch * out_h * out_w * module.out_channels * kernel_ops * 2)
        if module.bias is not None:
            flops += int(batch * out_h * out_w * module.out_channels)

    def linear_hook(module, inp, out):
        nonlocal flops
        x = inp[0]
        batch_ops = int(x.numel() / module.in_features)
        flops += int(batch_ops * module.in_features * module.out_features * 2)
        if module.bias is not None:
            flops += int(batch_ops * module.out_features)

    def norm_hook(module, inp, out):
        nonlocal flops
        flops += int(out.numel() * 2)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            handles.append(module.register_forward_hook(norm_hook))
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return int(flops)


def profiler_flops(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> int:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                model(inputs)
        total = sum(int(getattr(evt, "flops", 0) or 0) for evt in prof.key_averages())
        return int(total)
    except Exception:
        return 0


def benchmark_latency(model: nn.Module, inputs: torch.Tensor, device: torch.device, warmup: int, iters: int) -> dict[str, float | int | None]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    latency_ms = elapsed / max(1, iters) * 1000.0
    result: dict[str, float | int | None] = {
        "latency_ms": latency_ms,
        "fps": inputs.shape[0] * 1000.0 / latency_ms,
        "benchmark_warmup": int(warmup),
        "benchmark_iters": int(iters),
    }
    if device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        result["peak_memory_mb"] = None
    return result


def format_markdown(result: dict) -> str:
    keys = [
        "experiment",
        "deploy",
        "params_m",
        "flops_g",
        "latency_ms",
        "fps",
        "peak_memory_mb",
    ]
    header = "| " + " | ".join(keys) + " |"
    align = "| " + " | ".join(["---"] + ["---:" for _ in keys[1:]]) + " |"
    values = []
    for key in keys:
        value = result.get(key)
        if isinstance(value, float):
            values.append(f"{value:.4f}")
        elif value is None:
            values.append("")
        else:
            values.append(str(value))
    return "\n".join([header, align, "| " + " | ".join(values) + " |"])


def main():
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), args.cfg_options)
    seed_everything(int(cfg.seed), bool(cfg.deterministic))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    else:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = SPD(**cfg.model).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(normalize_state_dict(state_dict), strict=True)
    if args.deploy and hasattr(model.image_encoder, "switch_to_deploy"):
        model.eval()
        model.image_encoder.switch_to_deploy()
    model.eval()

    inputs = torch.randn(*args.input_size, device=device)
    params = count_parameters(model)
    prof_flops = profiler_flops(model, inputs, device)
    h_flops = hook_flops(model, inputs)
    flops = prof_flops if prof_flops > 0 else h_flops
    latency = benchmark_latency(model, inputs, device, args.warmup, args.iters)

    result = {
        "experiment": cfg.experiment_name,
        "config": args.config,
        "cfg_options": args.cfg_options or [],
        "device": str(device),
        "input_size": list(args.input_size),
        "deploy": bool(args.deploy),
        **params,
        "flops": int(flops),
        "flops_g": flops / 1e9,
        "flops_source": "torch_profiler" if prof_flops > 0 else "forward_hooks_conv_linear_norm",
        **latency,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(format_markdown(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
