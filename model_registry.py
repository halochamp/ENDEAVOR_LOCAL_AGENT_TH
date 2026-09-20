"""Declarative standalone model registry for Agent TH.

Native-AR APC behavior is capability-driven. Runtime code reads one model's
contract from model_registry.json instead of branching on repo IDs, so future
models are added as data while the APC launcher remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("model_registry.json")
SCHEMA_VERSION = 1
SUPPORTED_APC_TEMPLATE_POLICIES = frozenset({
    "qwen36_tool_call",
    "qwen35_tool_call",
    "qwen3_tool_call",
    "native_preserve",
})


class ModelRegistryError(ValueError):
    """Raised when the public model registry is unsafe or inconsistent."""


@dataclass(frozen=True)
class NativeAPCContract:
    enabled: bool
    exact_cache_entries: int
    template_policy: str


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    label: str
    role: str
    selectable: bool
    ram_warning_below_gib: int | None
    native_apc: NativeAPCContract


@dataclass(frozen=True)
class ModelRegistry:
    default_model: str
    models: tuple[ModelSpec, ...]

    @property
    def by_id(self) -> dict[str, ModelSpec]:
        return {spec.repo_id: spec for spec in self.models}

    def get(self, repo_id: str) -> ModelSpec:
        try:
            return self.by_id[str(repo_id)]
        except (KeyError, TypeError):
            raise ModelRegistryError(f"unknown model: {repo_id!r}") from None

    def role(self, name: str) -> ModelSpec:
        matches = [spec for spec in self.models if spec.role == name]
        if len(matches) != 1:
            raise ModelRegistryError(
                f"model role {name!r} must resolve to exactly one model"
            )
        return matches[0]

    @property
    def selectable_models(self) -> tuple[ModelSpec, ...]:
        return tuple(spec for spec in self.models if spec.selectable)


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRegistryError(f"{field} must be a non-empty string")
    return value.strip()


def _parse_model(raw: Any, *, index: int) -> ModelSpec:
    if not isinstance(raw, dict):
        raise ModelRegistryError(f"models[{index}] must be an object")

    repo_id = _require_text(raw.get("repo_id"), f"models[{index}].repo_id")
    label = _require_text(raw.get("label"), f"models[{index}].label")
    role = _require_text(raw.get("role"), f"models[{index}].role")
    selectable = raw.get("selectable")
    if type(selectable) is not bool:
        raise ModelRegistryError(f"models[{index}].selectable must be boolean")

    ram_warning = raw.get("ram_warning_below_gib")
    if ram_warning is not None and (
        type(ram_warning) is not int or ram_warning <= 0
    ):
        raise ModelRegistryError(
            f"models[{index}].ram_warning_below_gib must be null or positive integer"
        )

    apc = raw.get("native_apc")
    if not isinstance(apc, dict):
        raise ModelRegistryError(f"models[{index}].native_apc must be an object")
    enabled = apc.get("enabled")
    if type(enabled) is not bool:
        raise ModelRegistryError(
            f"models[{index}].native_apc.enabled must be boolean"
        )
    entries = apc.get("exact_cache_entries")
    policy = _require_text(
        apc.get("template_policy"),
        f"models[{index}].native_apc.template_policy",
    )
    if enabled:
        if entries != 2:
            raise ModelRegistryError(
                f"{repo_id} native APC requires exact_cache_entries=2"
            )
        if policy not in SUPPORTED_APC_TEMPLATE_POLICIES:
            raise ModelRegistryError(
                f"{repo_id} native APC template_policy must be one of "
                f"{sorted(SUPPORTED_APC_TEMPLATE_POLICIES)!r}"
            )
    elif type(entries) is not int or entries < 0:
        raise ModelRegistryError(
            f"{repo_id} disabled native APC requires non-negative exact_cache_entries"
        )

    return ModelSpec(
        repo_id=repo_id,
        label=label,
        role=role,
        selectable=selectable,
        ram_warning_below_gib=ram_warning,
        native_apc=NativeAPCContract(
            enabled=enabled,
            exact_cache_entries=int(entries),
            template_policy=policy,
        ),
    )


def load_registry(path: str | Path = REGISTRY_PATH) -> ModelRegistry:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"cannot read model registry: {exc}") from exc

    if not isinstance(payload, dict):
        raise ModelRegistryError("model registry root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ModelRegistryError(
            f"unsupported model registry schema: {payload.get('schema_version')!r}"
        )

    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ModelRegistryError("model registry requires a non-empty models list")
    models = tuple(_parse_model(raw, index=i) for i, raw in enumerate(raw_models))

    repo_ids = [spec.repo_id for spec in models]
    if len(repo_ids) != len(set(repo_ids)):
        raise ModelRegistryError("model registry contains duplicate repo_id entries")
    roles = [spec.role for spec in models]
    if len(roles) != len(set(roles)):
        raise ModelRegistryError("model registry contains duplicate role entries")

    default_model = _require_text(payload.get("default_model"), "default_model")
    by_id = {spec.repo_id: spec for spec in models}
    if default_model not in by_id:
        raise ModelRegistryError("default_model is not present in models")
    if not by_id[default_model].selectable:
        raise ModelRegistryError("default_model must be selectable")

    return ModelRegistry(default_model=default_model, models=models)


REGISTRY = load_registry()


def get_model_spec(model: str) -> ModelSpec:
    return REGISTRY.get(model)


def get_native_apc_contract(model: str) -> NativeAPCContract:
    return get_model_spec(model).native_apc


__all__ = [
    "ModelRegistry",
    "ModelRegistryError",
    "ModelSpec",
    "NativeAPCContract",
    "REGISTRY",
    "REGISTRY_PATH",
    "SUPPORTED_APC_TEMPLATE_POLICIES",
    "get_model_spec",
    "get_native_apc_contract",
    "load_registry",
]
