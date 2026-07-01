from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_SUPPORTED_MODULE_WARNING_TOKEN = "not in the model's supported LoRA target modules"
_TARGET_RESTRICTION_WARNING_TOKEN = "not in the deployment-time target_modules restriction"
_PACKED_RUNTIME_MODULE_ALIASES = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
    "linear_fc1": ("gate_proj", "up_proj"),
}


def _normalize_module_names(modules: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for module in modules:
        name = str(module).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _coerce_module_names(raw_modules: Any) -> list[str]:
    if isinstance(raw_modules, str):
        return _normalize_module_names(raw_modules.split(","))
    if isinstance(raw_modules, Iterable):
        return _normalize_module_names(raw_modules)
    return []


def _is_expected_packed_child_module(
    *,
    module_name: str,
    supported_modules: Iterable[Any],
) -> bool:
    module_suffix = str(module_name).rsplit(".", 1)[-1].strip()
    if not module_suffix:
        return False

    normalized_supported = set(_coerce_module_names(supported_modules))
    for packed_module, aliases in _PACKED_RUNTIME_MODULE_ALIASES.items():
        if module_suffix in aliases and packed_module in normalized_supported:
            return True
    return False


def classify_runtime_lora_warning_call(
    message: Any,
    args: tuple[Any, ...],
) -> tuple[str, str | None]:
    text = str(message)

    if _SUPPORTED_MODULE_WARNING_TOKEN in text:
        module_name = str(args[0]) if len(args) > 0 else "<unknown>"
        adapter_path = str(args[1]) if len(args) > 1 else "<unknown>"
        supported_modules = _coerce_module_names(args[2] if len(args) > 2 else [])
        if _is_expected_packed_child_module(
            module_name=module_name,
            supported_modules=supported_modules,
        ):
            return "ignore", None
        supported_display = ", ".join(supported_modules) or "<unknown>"
        return (
            "raise",
            "vLLM would ignore a LoRA module during adapter load. "
            f"module={module_name!r} adapter={adapter_path!r} "
            f"supported_runtime_modules=[{supported_display}]",
        )

    if _TARGET_RESTRICTION_WARNING_TOKEN in text:
        module_name = str(args[0]) if len(args) > 0 else "<unknown>"
        adapter_path = str(args[1]) if len(args) > 1 else "<unknown>"
        target_modules = _coerce_module_names(args[2] if len(args) > 2 else [])
        target_display = ", ".join(target_modules) or "<unknown>"
        return (
            "raise",
            "vLLM would ignore a LoRA module because of the deployment-time "
            f"target_modules restriction. module={module_name!r} "
            f"adapter={adapter_path!r} target_modules=[{target_display}]",
        )

    return "pass", None


def install_runtime_lora_warning_guard() -> None:
    from vllm.lora import worker_manager as vllm_worker_manager

    logger = vllm_worker_manager.logger
    if getattr(logger, "_arc3_runtime_lora_guard_installed", False):
        return

    original_warning_once = logger.warning_once

    def guarded_warning_once(message: Any, *args: Any, **kwargs: Any) -> Any:
        action, detail = classify_runtime_lora_warning_call(message, args)
        if action == "ignore":
            return None
        if action == "raise":
            raise RuntimeError(detail)
        return original_warning_once(message, *args, **kwargs)

    logger.warning_once = guarded_warning_once
    logger._arc3_runtime_lora_guard_installed = True

