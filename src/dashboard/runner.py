"""Run single-combination dashboard demo jobs against canonical artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from src.dashboard.paths import (
    EXPERIMENT_FIGURE_ROOT,
    PROJECT_ROOT,
    baseline_path,
    experiment_path,
    rel_path,
    validate_selection,
)


RUN_HISTORY = PROJECT_ROOT / "outputs" / "ui" / "run_history.jsonl"
TMP_CONFIG_DIR = PROJECT_ROOT / "outputs" / "ui" / "tmp_configs"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return rel_path(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k not in {"y_true", "y_pred"}}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _top_confusions(cm: np.ndarray, class_names: list[str], top_n: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            if i != j and count > 0:
                rows.append({"true": class_names[i], "pred": class_names[j], "count": count})
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows[:top_n]


def _candidate_config(dataset: str, seed: int, model: str, method: str | None, mode: str | None) -> Path:
    base = PROJECT_ROOT / "configs" / "generated" / dataset / f"seed_{seed}" / model
    if method and mode:
        path = base / method / f"{mode}.yaml"
        if path.exists():
            return path
    path = base / "baseline.yaml"
    if path.exists():
        return path
    return PROJECT_ROOT / "configs" / "default.yaml"


def _target_dir(dataset: str, seed: int, model: str, job: str, method: str | None) -> Path:
    if job in {"train_baseline", "evaluate_baseline"}:
        return baseline_path(dataset, seed, model)
    mode = "fixed_attacker" if job == "defense_eval_fixed" else "retrain_attacker"
    if method is None:
        raise ValueError("method is required for defense jobs")
    return experiment_path(dataset, seed, model, method, mode)


def _model_dir(dataset: str, seed: int, model: str, job: str, method: str | None) -> Path:
    root = PROJECT_ROOT / "outputs" / "models" / dataset / f"seed_{seed}" / model
    if job in {"train_baseline", "evaluate_baseline"}:
        return root / "baseline"
    mode = "fixed_attacker" if job == "defense_eval_fixed" else "retrain_attacker"
    if method is None:
        raise ValueError("method is required for defense jobs")
    return root / method / mode


def _figure_dir(dataset: str, seed: int, model: str, job: str, method: str | None) -> Path:
    if job in {"train_baseline", "evaluate_baseline"}:
        return EXPERIMENT_FIGURE_ROOT / dataset / f"seed_{seed}" / model / "baseline"
    mode = "fixed_attacker" if job == "defense_eval_fixed" else "retrain_attacker"
    if method is None:
        raise ValueError("method is required for defense jobs")
    return EXPERIMENT_FIGURE_ROOT / dataset / f"seed_{seed}" / model / method / mode


def _is_complete(target: Path, job: str, model_path: Path) -> bool:
    if job == "train_baseline":
        return model_path.exists()
    if job == "evaluate_baseline":
        return (target / "baseline_metrics.json").exists() and (target / "baseline_confusion.json").exists()
    return (target / "metrics.json").exists() and (target / "confusion.json").exists()


def build_demo_config(
    dataset: str,
    seed: int,
    model: str,
    job: str,
    method: str | None = None,
    max_epochs: int = 5,
    batch_size: int | None = None,
    device: str = "auto",
) -> Path:
    mode = "fixed_attacker" if job == "defense_eval_fixed" else "retrain_attacker" if job == "defense_eval_retrain" else None
    cfg_path = _candidate_config(dataset, seed, model, method, mode)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {rel_path(cfg_path)}")
    raw = copy.deepcopy(raw)

    paths = raw.setdefault("paths", {})
    paths["processed_dir"] = f"data/processed/{dataset}/seed_{seed}"
    if method:
        paths["defended_dir"] = f"data/defended/{dataset}/seed_{seed}/{method}"
    else:
        paths["defended_dir"] = f"data/defended/{dataset}/seed_{seed}/adaptive_ldp"
    paths["models_dir"] = rel_path(_model_dir(dataset, seed, model, job, method))
    paths["reports_dir"] = rel_path(_target_dir(dataset, seed, model, job, method))
    paths["figures_dir"] = rel_path(_figure_dir(dataset, seed, model, job, method))
    paths["defense_dir"] = rel_path(_target_dir(dataset, seed, model, job, method))

    raw.setdefault("experiment", {})["random_seed"] = int(seed)
    train = raw.setdefault("train", {})
    train["model_type"] = model
    train["num_epochs"] = int(max_epochs)
    train["early_stopping_patience"] = min(int(train.get("early_stopping_patience", 10)), int(max_epochs))
    train["device"] = device
    if batch_size:
        train["batch_size"] = int(batch_size)
    raw.setdefault("evaluate", {})["device"] = device
    if batch_size:
        raw["evaluate"]["batch_size"] = int(batch_size)
    if method:
        defense = raw.setdefault("defense", {})
        defense["method"] = method
        defense["mode"] = mode
    raw.setdefault("defense_eval", {})["retrained_model_name"] = f"best_{model}_defended_retrain.pt"

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{dataset}.seed_{seed}.{model}.{job}" + (f".{method}" if method else "")
    out = TMP_CONFIG_DIR / f"{suffix}.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def _load_cfg(config_path: Path):
    from src.core.config import ExperimentConfig

    return ExperimentConfig.from_yaml(config_path, project_root=PROJECT_ROOT)


def _processed_paths(dataset: str, seed: int) -> tuple[Path, Path, Path]:
    processed = PROJECT_ROOT / "data" / "processed" / dataset / f"seed_{seed}"
    return processed, processed / "sequences.npz", processed / "mlp_features.npz"


def _evaluate_model(config_path: Path, model_path: Path, model_type: str) -> dict[str, Any]:
    import numpy as np

    from src.evaluation.evaluator import evaluate_on_arrays, load_model_from_checkpoint
    from src.core.utils import get_torch_device

    cfg = _load_cfg(config_path)
    ev = cfg.nested("evaluate")
    device = get_torch_device(str(ev.get("device") or "auto"))
    model, ckpt = load_model_from_checkpoint(model_path, device)
    class_names = list(ckpt["class_names"])
    processed = cfg.path("paths", "processed_dir")
    if model_type == "lstm":
        data = np.load(processed / "sequences.npz")
        x_test, y_test = data["X_test"], data["y_test"]
    else:
        data = np.load(processed / "mlp_features.npz")
        x_test, y_test = data["X_test"], data["y_test"]
    return evaluate_on_arrays(
        model,
        x_test,
        y_test,
        model_type,
        class_names,
        device,
        int(ev.get("batch_size", 128)),
        int(ev.get("num_workers", 0)),
    ) | {"class_names": class_names}


def _write_eval_artifacts(target: Path, metrics: dict[str, Any], prefix: str, model_type: str) -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    class_names = [str(x) for x in metrics.get("class_names", [])]
    payload = {
        "split": "test",
        "model_type": model_type,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "per_class_recall": metrics["per_class_recall"],
        "confusion_matrix": cm.tolist(),
        "top_confusions": _top_confusions(cm, class_names),
        "class_names": class_names,
    }
    if prefix == "baseline":
        _write_json(target / "baseline_metrics.json", payload)
        _write_json(target / "baseline_confusion.json", payload)
        _write_text(target / "baseline_classification_report.txt", str(metrics["classification_report"]))
        _write_json(target / "baseline_trace.json", {"generated_at": _now(), "artifact_type": "baseline_evaluation"})
    else:
        _write_json(target / "metrics.json", payload)
        _write_json(target / "confusion.json", payload)
        _write_text(target / "classification_report.txt", str(metrics["classification_report"]))
        _write_json(target / "trace.json", {"generated_at": _now(), "artifact_type": "defense_evaluation"})


def _write_manifest(
    target: Path,
    *,
    dataset: str,
    seed: int,
    model: str,
    job: str,
    method: str | None,
    mode: str | None,
    config_path: Path,
    model_path: Path | None,
    status: str,
) -> None:
    _write_json(
        target / "source_manifest.json",
        {
            "generated_at": _now(),
            "dataset": dataset,
            "seed": int(seed),
            "model": model,
            "method": method,
            "mode": mode,
            "role": job,
            "config_path": rel_path(config_path),
            "model_path": rel_path(model_path) if model_path else None,
            "status": status,
            "canonical_path": rel_path(target),
            "note": "Generated by the dashboard single-combination demo runner; no data import or full-matrix run was executed.",
        },
    )


def execute_dashboard_job(
    dataset: str,
    seed: int,
    model: str,
    job: str,
    method: str | None = None,
    max_epochs: int = 5,
    batch_size: int | None = None,
    device: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    messages: list[str] = []

    def emit(message: str) -> None:
        messages.append(message)
        print(message, flush=True)

    if job not in {"train_baseline", "evaluate_baseline", "defense_eval_fixed", "defense_eval_retrain"}:
        raise ValueError(f"Unsupported job: {job}")
    ok, msg = validate_selection(dataset, seed, model, method if job.startswith("defense") else None)
    if not ok:
        raise ValueError(msg)
    if dataset == "cooja":
        raise ValueError("Dashboard demo runner does not run Cooja simulation jobs.")

    processed, seq_path, mlp_path = _processed_paths(dataset, seed)
    if not processed.exists() or not seq_path.exists() or (model == "mlp" and not mlp_path.exists()):
        raise FileNotFoundError(f"Processed data is missing for {dataset}/seed_{seed}: {rel_path(processed)}")

    target = _target_dir(dataset, seed, model, job, method)
    model_dir = _model_dir(dataset, seed, model, job, method)
    model_path = model_dir / f"best_{model}.pt"
    if job.startswith("defense") and method:
        mode = "fixed_attacker" if job == "defense_eval_fixed" else "retrain_attacker"
        defended = PROJECT_ROOT / "data" / "defended" / dataset / f"seed_{seed}" / method
        if not defended.exists():
            raise FileNotFoundError(f"Defended data is missing: {rel_path(defended)}")
        baseline_model = PROJECT_ROOT / "outputs" / "models" / dataset / f"seed_{seed}" / model / "baseline" / f"best_{model}.pt"
        if job == "defense_eval_fixed":
            model_path = baseline_model
        else:
            model_path = model_dir / f"best_{model}_defended_retrain.pt"
    else:
        mode = None

    if _is_complete(target, job, model_path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifacts at {rel_path(target)}. Enable overwrite to run this job.")

    config_path = build_demo_config(dataset, seed, model, job, method, max_epochs, batch_size, device)
    target.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    emit(f"CONFIG_PREPARED {rel_path(config_path)}")

    if job == "train_baseline":
        from src.training.trainer import run_training

        emit(f"TRAINING_STARTED model={model} max_epochs={max_epochs}")
        cfg = _load_cfg(config_path)
        run_training(
            cfg,
            model_type=model,
            override_model_path=model_path,
            curve_output_path=_figure_dir(dataset, seed, model, job, method) / "train_curve.png",
            history_output_path=target / "train_history.json",
        )
        _write_json(target / "baseline_trace.json", {"generated_at": _now(), "artifact_type": "baseline_training", "max_epochs": max_epochs})
        _write_manifest(
            target,
            dataset=dataset,
            seed=seed,
            model=model,
            job=job,
            method=None,
            mode=None,
            config_path=config_path,
            model_path=model_path,
            status="success",
        )
    elif job == "evaluate_baseline":
        emit("EVALUATION_STARTED baseline")
        baseline_model = PROJECT_ROOT / "outputs" / "models" / dataset / f"seed_{seed}" / model / "baseline" / f"best_{model}.pt"
        if not baseline_model.exists():
            raise FileNotFoundError(f"Missing baseline model: {rel_path(baseline_model)}")
        metrics = _evaluate_model(config_path, baseline_model, model)
        _write_eval_artifacts(target, metrics, "baseline", model)
        _write_manifest(
            target,
            dataset=dataset,
            seed=seed,
            model=model,
            job=job,
            method=None,
            mode=None,
            config_path=config_path,
            model_path=baseline_model,
            status="success",
        )
    else:
        from src.evaluation.defense_evaluator import compute_fixed_attacker_metrics, run_defense_evaluation
        from src.evaluation.evaluator import evaluate_on_arrays, load_model_from_checkpoint
        from src.core.utils import get_torch_device

        cfg = _load_cfg(config_path)
        emit(f"DEFENSE_EVALUATION_STARTED method={method} mode={mode}")
        if job == "defense_eval_fixed":
            baseline_model = PROJECT_ROOT / "outputs" / "models" / dataset / f"seed_{seed}" / model / "baseline" / f"best_{model}.pt"
            run_defense_evaluation(cfg, mode="fixed_attacker", model_path=baseline_model, skip_pipeline=True)
            pair = compute_fixed_attacker_metrics(cfg, baseline_model)
            metrics = pair["defended"] | {"class_names": pair["class_names"]}
        else:
            run_defense_evaluation(cfg, mode="retrain_attacker", model_path=None, skip_pipeline=True)
            ev = cfg.nested("evaluate")
            device_obj = get_torch_device(str(ev.get("device") or "auto"))
            retrained_model = model_dir / f"best_{model}_defended_retrain.pt"
            model_obj, ckpt = load_model_from_checkpoint(retrained_model, device_obj)
            class_names = list(ckpt["class_names"])
            defended = cfg.path("paths", "defended_dir")
            if model == "lstm":
                data = np.load(defended / "defended_sequences.npz")
                x_test, y_test = data["X_test"], data["y_test"]
            else:
                data = np.load(defended / "defended_mlp_features.npz")
                x_test, y_test = data["X_test"], data["y_test"]
            metrics = evaluate_on_arrays(
                model_obj,
                x_test,
                y_test,
                model,
                class_names,
                device_obj,
                int(ev.get("batch_size", 128)),
                int(ev.get("num_workers", 0)),
            ) | {"class_names": class_names}
            model_path = retrained_model
        _write_eval_artifacts(target, metrics, "defense", model)
        _write_manifest(
            target,
            dataset=dataset,
            seed=seed,
            model=model,
            job=job,
            method=method,
            mode=mode,
            config_path=config_path,
            model_path=model_path,
            status="success",
        )

    emit(f"WRITING_ARTIFACTS {rel_path(target)}")
    emit("DONE")
    return {
        "status": "success",
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "method": method,
        "mode": mode,
        "job": job,
        "output_path": rel_path(target),
        "model_path": rel_path(model_path) if model_path else None,
        "config_path": rel_path(config_path),
        "stdout_tail": "\n".join(messages[-80:]),
    }


def parse_epoch_progress(line: str, max_epochs: int) -> int | None:
    match = re.search(r"Epoch\s+0*(\d+)", line)
    if not match:
        return None
    epoch = max(1, int(match.group(1)))
    return min(80, 20 + int((epoch / max(max_epochs, 1)) * 60))


def stream_subprocess(command: list[str], cwd: Path = PROJECT_ROOT) -> Iterator[dict[str, Any]]:
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield {"type": "line", "text": line.rstrip("\n")}
    rc = proc.wait()
    yield {"type": "returncode", "returncode": rc}


def run_dashboard_job(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    stdout_tail: list[str] = []
    stderr_tail = ""
    status = "success"
    result: dict[str, Any] = {}
    try:
        result = execute_dashboard_job(
            dataset=args.dataset,
            seed=int(args.seed),
            model=args.model,
            job=args.job,
            method=args.method,
            max_epochs=int(args.max_epochs),
            batch_size=args.batch_size,
            device=args.device,
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:
        status = "failed"
        stderr_tail = str(exc)
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        result = {
            "status": status,
            "dataset": args.dataset,
            "seed": int(args.seed),
            "model": args.model,
            "method": args.method,
            "mode": "fixed_attacker" if args.job == "defense_eval_fixed" else "retrain_attacker" if args.job == "defense_eval_retrain" else None,
            "job": args.job,
            "output_path": rel_path(_target_dir(args.dataset, int(args.seed), args.model, args.job, args.method)),
        }
    duration = time.time() - started
    result["status"] = status
    result["duration_seconds"] = round(duration, 3)
    result["command"] = " ".join([sys.executable, *sys.argv])
    result["stdout_tail"] = result.get("stdout_tail") or "\n".join(stdout_tail[-80:])
    result["stderr_tail"] = stderr_tail[-12000:]
    write_run_history(result)
    return result


def write_run_history(record: dict[str, Any]) -> None:
    RUN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _now(), **_jsonable(record)}
    with RUN_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one canonical dashboard training/evaluation job.")
    parser.add_argument("--dataset", choices=["mock", "uci_har", "kasteren", "casas_hh101"], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], required=True)
    parser.add_argument("--model", choices=["lstm", "mlp"], required=True)
    parser.add_argument("--job", choices=["train_baseline", "evaluate_baseline", "defense_eval_fixed", "defense_eval_retrain"], required=True)
    parser.add_argument("--method", choices=["adaptive_ldp", "ldp", "noise"], default=None)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser
