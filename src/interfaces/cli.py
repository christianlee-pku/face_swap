import argparse
from pathlib import Path
from typing import Optional

from runners.base_runner import build_runner
from utils.config import prepare_run
from data.preprocess_lfw import run_preprocess
from data.validate_manifest import load_manifest, validate_items, validate_splits


def cmd_train(args: argparse.Namespace) -> None:
    ctx = prepare_run(Path(args.config), Path(args.work_dir) if args.work_dir else None)
    runner = build_runner(ctx["work_dir"], ctx["config"], ctx.get("config_path"), ctx.get("env_hash", ""))
    runner.train()


def cmd_eval(args: argparse.Namespace) -> None:
    ctx = prepare_run(Path(args.config), Path(args.work_dir) if args.work_dir else None)
    runner = build_runner(ctx["work_dir"], ctx["config"], ctx.get("config_path"), ctx.get("env_hash", ""))
    metrics = runner.evaluate()
    print(metrics)


def cmd_infer(args: argparse.Namespace) -> None:
    from pathlib import Path
    import yaml
    from PIL import Image
    import torch
    from torchvision import transforms as T
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    infer_cfg = cfg.get("infer", {})
    sources = infer_cfg.get("sources", [])
    targets = infer_cfg.get("targets", [])
    output_dir = Path(infer_cfg.get("output_dir", "work_dirs/face_swap/infer-samples"))
    ckpt_path = infer_cfg.get("checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = prepare_run(Path(args.config), Path(args.work_dir) if args.work_dir else None)
    runner = build_runner(ctx["work_dir"], ctx["config"], ctx.get("config_path"), ctx.get("env_hash", ""))
    model_cfg = ctx["config"].get("model", {})
    if torch is None:
        raise RuntimeError("torch is required for inference")
    from registry import MODELS as MODELS_REG
    model = MODELS_REG.build(model_cfg)
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict, strict=False)
        runner.logger.info("Loaded checkpoint for infer: %s", ckpt_path)
    model.eval()
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()
    device = torch.device("cpu")
    model.to(device)
    pairs = list(zip(sources, targets))
    if not pairs:
        runner.logger.warning("No inference pairs defined in config.infer.sources/targets")
        return
    for src, tgt in pairs:
        src_img = Image.open(src).convert("RGB")
        tgt_img = Image.open(tgt).convert("RGB")
        src_tensor = to_tensor(src_img).unsqueeze(0).to(device)
        tgt_tensor = to_tensor(tgt_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(src_tensor, tgt_tensor)
            pred = outputs.get("output", tgt_tensor)
        pred = pred.clamp(0, 1).cpu().squeeze(0)
        out_img = to_pil(pred)
        out_name = f"{Path(src).stem}_to_{Path(tgt).stem}.png"
        out_path = output_dir / out_name
        out_img.save(out_path)
        runner.logger.info("Saved inference output to %s", out_path)


def cmd_export(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    export_cfg = cfg.get("export", {})
    ckpt_path = Path(export_cfg.get("checkpoint", ""))
    export_dir = Path(export_cfg.get("export_dir", "work_dirs/exports/placeholder"))
    fmt = export_cfg.get("format", "onnx")
    opset = export_cfg.get("opset", 14)
    input_size = export_cfg.get("input_size", [1, 3, 160, 160])
    export_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "export_dir": str(export_dir),
        "format": fmt,
        "opset": opset,
        "input_size": input_size,
        "status": "skipped",
        "error": "",
    }

    success = False
    onnx_path = export_dir / "model.onnx"
    try:
        import torch
        from registry import MODELS as MODELS_REG

        model_cfg = cfg.get("model", {})
        model = MODELS_REG.build(model_cfg)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            state_dict = state.get("model", state)
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        dummy_src = torch.randn(*input_size)
        dummy_tgt = torch.randn(*input_size)

        class Wrapper(torch.nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(self, src, tgt):
                out = self.core(src, tgt)
                return out.get("output", src)

        wrapped = Wrapper(model)
        torch.onnx.export(
            wrapped,
            (dummy_src, dummy_tgt),
            onnx_path,
            input_names=["source", "target"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
        )
        success = True
        meta["status"] = "exported"
        meta["onnx_path"] = str(onnx_path)
    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = str(e)

    (export_dir / "export.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True))
    if ckpt_path and ckpt_path.exists():
        import shutil

        shutil.copy2(ckpt_path, export_dir / ckpt_path.name)
    print(f"Export status: {meta['status']} (details in {export_dir/'export.meta.json'})")


def cmd_trt(args: argparse.Namespace) -> None:
    import json
    import subprocess
    from pathlib import Path
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    trt_cfg = cfg.get("export", {})
    onnx_path = Path(trt_cfg.get("onnx_path", ""))
    engine_path = Path(trt_cfg.get("engine_path", ""))
    trtexec = trt_cfg.get("trtexec_path", "trtexec")
    precision = trt_cfg.get("precision", "fp16")
    workspace = trt_cfg.get("workspace_size", 2048)
    max_batch = trt_cfg.get("max_batch_size", 1)
    dynamic = trt_cfg.get("dynamic_shapes", False)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace}",
        f"--minShapes=source:1x3x160x160,target:1x3x160x160" if dynamic else "",
        f"--optShapes=source:1x3x160x160,target:1x3x160x160" if dynamic else "",
        f"--maxShapes=source:{max_batch}x3x160x160,target:{max_batch}x3x160x160" if dynamic else "",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
    cmd = [c for c in cmd if c]  # drop empties

    meta = {
        "config": str(args.config),
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "precision": precision,
        "workspace_size": workspace,
        "max_batch_size": max_batch,
        "dynamic_shapes": dynamic,
        "cmd": cmd,
        "status": "pending",
        "error": "",
    }
    try:
        subprocess.run(cmd, check=True)
        meta["status"] = "exported"
    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = str(e)
    (engine_path.parent / "trt.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True))
    print(f"TensorRT status: {meta['status']} (details in {engine_path.parent/'trt.meta.json'})")


def cmd_prepare_data(args: argparse.Namespace) -> None:
    # Load config first
    from pathlib import Path
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    # Apply overrides if provided
    cfg["download"] = args.download if args.download else cfg.get("download", False)
    for key in ["dataset", "raw_dir", "proc_dir", "manifest", "train_ratio", "val_ratio", "version"]:
        override = getattr(args, key, None)
        if override is not None:
            cfg[key] = override
    run_preprocess(
        download=cfg.get("download", False),
        dataset=cfg.get("dataset", "ashishpatel26/lfw-dataset"),
        raw_dir=cfg.get("raw_dir", "data/lfw/raw"),
        proc_dir=cfg.get("proc_dir", "data/lfw/processed"),
        manifest=cfg.get("manifest", "data/lfw/manifest.json"),
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        version=cfg.get("version", "1.0.0"),
    )


def cmd_benchmark_edge(args: argparse.Namespace) -> None:
    from ..exporters.benchmarks import benchmark_edge
    ctx = prepare_run(Path(args.config), Path(args.work_dir) if args.work_dir else None)
    results = benchmark_edge(ctx["config"], Path(args.checkpoint), Path(args.export_dir), target=args.target)
    print(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face swap CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_validate = subparsers.add_parser("validate-manifest")
    p_validate.add_argument("--manifest", required=True)
    p_validate.add_argument("--processed-dir", required=True)
    p_validate.set_defaults(func=cmd_validate_manifest)

    p_data = subparsers.add_parser("prepare-data")
    p_data.add_argument("--config", required=True)
    p_data.add_argument("--download", action="store_true", help="Override config to force download")
    p_data.add_argument("--dataset", help="Override dataset ref")
    p_data.add_argument("--raw-dir", help="Override raw dir")
    p_data.add_argument("--proc-dir", help="Override processed dir")
    p_data.add_argument("--manifest", help="Override manifest path")
    p_data.add_argument("--train-ratio", type=float, help="Override train split ratio")
    p_data.add_argument("--val-ratio", type=float, help="Override val split ratio")
    p_data.add_argument("--version", help="Override manifest version")
    p_data.set_defaults(func=cmd_prepare_data)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--work-dir", required=False)
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("eval")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--work-dir", required=False)
    p_eval.set_defaults(func=cmd_eval)

    p_infer = subparsers.add_parser("infer")
    p_infer.add_argument("--config", required=True)
    p_infer.add_argument("--work-dir", required=False)
    p_infer.set_defaults(func=cmd_infer)

    p_export = subparsers.add_parser("export")
    p_export.add_argument("--config", required=True)
    p_export.add_argument("--work-dir", required=False)
    p_export.set_defaults(func=cmd_export)

    p_trt = subparsers.add_parser("trt")
    p_trt.add_argument("--config", required=True)
    p_trt.add_argument("--work-dir", required=False)
    p_trt.set_defaults(func=cmd_trt)

    p_bench = subparsers.add_parser("benchmark-edge")
    p_bench.add_argument("--config", required=True)
    p_bench.add_argument("--checkpoint", required=True)
    p_bench.add_argument("--export-dir", required=True)
    p_bench.add_argument("--target", default="jetson")
    p_bench.add_argument("--work-dir", required=False)
    p_bench.set_defaults(func=cmd_benchmark_edge)

    return parser

def cmd_validate_manifest(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    processed_dir = Path(args.processed_dir)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
    manifest = load_manifest(manifest_path)
    missing, bad_checksum = validate_items(manifest, processed_dir)
    splits_ok = validate_splits(manifest)
    if missing:
        print("Missing files:", missing)
    if bad_checksum:
        print("Checksum mismatches:", bad_checksum)
    if not splits_ok:
        print("Split validation failed.")
    if not missing and not bad_checksum and splits_ok:
        print("Manifest validation PASSED")

def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
