"""Typed, fail-closed model-slot registry (forked from Agent MAX VLM).

Agent TH ships native-AR slots only: model_slots.json lists no DFlash / MTP accelerator slot, so none of the draft code below is
reached here. It is kept as is so the two registries stay mergeable; a slot pins the exact Hugging Face commit (`revision`) and
`preflight_slot` proves that commit is fully downloaded before the Model page calls it installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REGISTRY_PATH = Path(__file__).with_name("model_slots.json")
SCHEMA_VERSION = 2
SUPPORTED_ADAPTERS = frozenset({
    "mlx_vlm_dflash_v1",
    "mlx_vlm_dflash2",
    "mlx_vlm_mtp",
})
_HEX40 = set("0123456789abcdef")
_DFLASH_BACKENDS = frozenset({"dflash", "dflash2"})
_DFLASH_DISPLAY_NAMES = {"dflash": "DFlash", "dflash2": "DFlash2"}
_DFLASH_QUANTIZED_MAX_BLOCK = 5
_MTP_ADAPTER = "mlx_vlm_mtp"
_MTP_BACKEND = "mtp"
_MTP_DISPLAY_NAME = "MTP"
_ACCELERATION_MODES = frozenset({"off", "dflash", "mtp"})
_NATIVE_APC_TEMPLATE_POLICIES = frozenset({
    "qwen36_tool_call",
    "qwen36_heretic_tool_call",
    "qwen35_tool_call",
    "qwen3_tool_call",
    "native_preserve",
    "qwen38_dflash2_tool_choice",
})


class ModelSlotError(ValueError):
    """Raised when a slot registry or selection is unsafe to use."""


@dataclass(frozen=True)
class TargetSpec:
    repo_id: str
    revision: str


@dataclass(frozen=True)
class DraftSpec:
    engine: str
    repo_id: str
    revision: str
    load_bits: int | None
    block_size: int


@dataclass(frozen=True)
class ModelSlot:
    slot_id: str
    label: str
    selectable: bool
    kind: str
    target: TargetSpec
    draft: DraftSpec | None
    routing: Mapping[str, str]
    runtime: Mapping[str, Any]
    digest: str

    @property
    def model(self) -> str:
        return self.target.repo_id

    @property
    def dflash_backend(self) -> str | None:
        """Return the normalized speculative backend kind for this slot."""
        if self.draft is None:
            return None
        configured = self.runtime.get("draft_kind")
        if configured in _DFLASH_BACKENDS:
            return str(configured)
        return {
            "mlx_vlm_dflash_v1": "dflash",
            "mlx_vlm_dflash2": "dflash2",
        }.get(self.draft.engine)

    @property
    def mtp_backend(self) -> str | None:
        """Return the normalized native MTP backend for this slot."""
        if self.draft is not None and self.draft.engine == _MTP_ADAPTER:
            return _MTP_BACKEND
        return None

    @property
    def mtp_display_name(self) -> str | None:
        if self.mtp_backend is None:
            return None
        configured = self.runtime.get("mtp_display_name")
        return str(configured).strip() if configured else _MTP_DISPLAY_NAME

    @property
    def acceleration_mode(self) -> str:
        """Return the mutually-exclusive route represented by this slot."""
        if self.dflash_backend is not None:
            return "dflash"
        if self.mtp_backend is not None:
            return "mtp"
        return "off"

    @property
    def acceleration_display_name(self) -> str:
        if self.acceleration_mode == "dflash":
            return self.dflash_display_name or "DFlash"
        if self.acceleration_mode == "mtp":
            return self.mtp_display_name or _MTP_DISPLAY_NAME
        return "Off"

    @property
    def dflash_display_name(self) -> str | None:
        backend = self.dflash_backend
        return _DFLASH_DISPLAY_NAMES.get(backend) if backend else None


@dataclass(frozen=True)
class ModelSlotRegistry:
    schema_version: int
    default_slot_id: str
    legacy_model_defaults: Mapping[str, str]
    slots: tuple[ModelSlot, ...]

    @property
    def by_id(self) -> dict[str, ModelSlot]:
        return {slot.slot_id: slot for slot in self.slots}

    def get(self, slot_id: str) -> ModelSlot:
        try:
            return self.by_id[str(slot_id)]
        except (KeyError, TypeError):
            raise ModelSlotError(f"unknown model slot: {slot_id!r}") from None

    def for_model(self, model: str) -> ModelSlot:
        model = str(model or "")
        slot_id = self.legacy_model_defaults.get(model)
        if slot_id is None:
            matches = [slot for slot in self.slots if slot.model == model]
            if len(matches) != 1:
                raise ModelSlotError(f"unknown or ambiguous legacy model: {model!r}")
            return matches[0]
        return self.get(slot_id)

    def resolve(self, *, slot_id: str | None = None, model: str | None = None) -> ModelSlot:
        if slot_id is not None and model is not None:
            slot = self.get(slot_id)
            # The model field is only a compatibility target mirror.  It must
            # agree with the authoritative slot target, but legacy defaults
            # cannot disambiguate two slots sharing that target.
            if slot.model != str(model or ""):
                raise ModelSlotError("conflicting slot_id and model")
            return slot
        if slot_id is not None:
            return self.get(slot_id)
        if model is not None:
            return self.for_model(model)
        return self.get(self.default_slot_id)

    @staticmethod
    def _same_target(left: ModelSlot, right: ModelSlot) -> bool:
        return (
            left.model == right.model
            and left.target.revision == right.target.revision
        )

    def _target_variants(self, slot: ModelSlot) -> list[ModelSlot]:
        return [candidate for candidate in self.slots if self._same_target(slot, candidate)]

    @staticmethod
    def _variant_sort_key(slot: ModelSlot, default_key: str) -> tuple[bool, bool, str]:
        # A future registry can mark one variant as the configured default. If
        # it does not, prefer a selectable route and keep registry order stable
        # through the final slot_id tie-breaker.
        return (
            not bool(slot.runtime.get(default_key, False)),
            not slot.selectable,
            slot.slot_id,
        )

    def dflash_variant(self, slot: ModelSlot) -> ModelSlot | None:
        """Resolve the configured DFlash route for a target, if one exists."""
        if slot.dflash_backend is not None:
            return slot
        candidates = [
            candidate for candidate in self._target_variants(slot)
            if candidate.dflash_backend is not None
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda candidate: self._variant_sort_key(candidate, "dflash_default"),
        )[0]

    def ar_variant(self, slot: ModelSlot) -> ModelSlot | None:
        """Resolve the native AR route for the same target."""
        if slot.draft is None:
            return slot
        candidates = [
            candidate for candidate in self._target_variants(slot)
            if candidate.draft is None and candidate.routing.get("text") == "native_ar"
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda candidate: self._variant_sort_key(candidate, "ar_default"),
        )[0]

    def mtp_variant(self, slot: ModelSlot) -> ModelSlot | None:
        """Resolve the configured native MTP route for a target, if any."""
        if slot.mtp_backend is not None:
            return slot
        candidates = [
            candidate for candidate in self._target_variants(slot)
            if candidate.mtp_backend is not None
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda candidate: self._variant_sort_key(candidate, "mtp_default"),
        )[0]

    def dflash_metadata(self, slot: ModelSlot) -> dict[str, Any]:
        """Return normalized capability and requested-route metadata for a slot."""
        dflash = self.dflash_variant(slot)
        ar = self.ar_variant(slot)
        supported = dflash is not None
        return {
            "dflash_supported": supported,
            "dflash_backend": dflash.dflash_backend if dflash else None,
            "dflash_display_name": dflash.dflash_display_name if dflash else None,
            # ``selectable`` is the production gate. A known but gated route
            # remains visible as a capability, but is not activatable.
            "dflash_available": bool(dflash and dflash.selectable),
            "dflash_slot_id": dflash.slot_id if dflash else None,
            "dflash_draft_model": (
                dflash.draft.repo_id if dflash and dflash.draft else None
            ),
            "dflash_ar_slot_id": ar.slot_id if ar else None,
            "dflash_enabled": bool(dflash and slot.slot_id == dflash.slot_id),
        }

    @staticmethod
    def _global_apc_enabled(slot: ModelSlot) -> bool | None:
        """Return the process-wide APC contract for one selected route."""
        explicit = slot.runtime.get("apc_enabled")
        if isinstance(explicit, bool):
            return explicit
        mode = str(slot.runtime.get("apc") or "").strip().lower()
        if mode == "disabled_until_verified":
            return False
        if mode == "native":
            return True
        return None

    def route_for_dflash(self, slot: ModelSlot, enabled: bool) -> ModelSlot:
        """Resolve AR or DFlash for ``slot`` without changing its target."""
        route = self.dflash_variant(slot) if enabled else self.ar_variant(slot)
        if route is None:
            state = "DFlash" if enabled else "native AR"
            raise ModelSlotError(f"no {state} route is configured for target {slot.model!r}")
        if enabled and not route.selectable:
            raise ModelSlotError(
                f"DFlash route is not selectable until verified: {route.slot_id!r}"
            )
        if not enabled and not route.selectable:
            raise ModelSlotError(f"native AR route is not selectable: {route.slot_id!r}")
        return route

    def acceleration_metadata(self, slot: ModelSlot) -> dict[str, Any]:
        """Return one normalized, target-level acceleration contract.

        The registry knows route identity and the durable production gate. The
        runtime/provider layer may further reduce ``available``; callers should
        overlay that dynamic result rather than infer anything from model names.
        """
        ar = self.ar_variant(slot)
        dflash = self.dflash_variant(slot)
        mtp = self.mtp_variant(slot)

        def option(mode: str, route: ModelSlot | None, display_name: str) -> dict[str, Any]:
            return {
                "mode": mode,
                "display_name": display_name,
                "backend": None if route is None or mode == "off" else (
                    route.dflash_backend if mode == "dflash" else route.mtp_backend
                ),
                "supported": route is not None,
                "available": bool(route is not None and route.selectable),
                "selectable": bool(route is not None and route.selectable),
                "slot_id": route.slot_id if route is not None else None,
                "model": route.model if route is not None else slot.model,
                "target_revision": route.target.revision if route is not None else slot.target.revision,
                "draft_model": (
                    route.draft.repo_id if route is not None and route.draft is not None else None
                ),
                "identity": (
                    f"{route.slot_id}@{route.digest}" if route is not None else None
                ),
            }

        options = [
            option("off", ar, "Off"),
            option("dflash", dflash, dflash.dflash_display_name if dflash else "DFlash"),
            option("mtp", mtp, mtp.mtp_display_name if mtp else _MTP_DISPLAY_NAME),
        ]
        return {
            "acceleration_mode": slot.acceleration_mode,
            "acceleration_backend": slot.acceleration_display_name,
            "acceleration_options": options,
            "dflash_supported": dflash is not None,
            "dflash_available": bool(dflash and dflash.selectable),
            "dflash_backend": dflash.dflash_backend if dflash else None,
            "dflash_display_name": dflash.dflash_display_name if dflash else None,
            "dflash_slot_id": dflash.slot_id if dflash else None,
            "dflash_draft_model": dflash.draft.repo_id if dflash and dflash.draft else None,
            "mtp_supported": mtp is not None,
            "mtp_available": bool(mtp and mtp.selectable),
            "mtp_backend": mtp.mtp_backend if mtp else None,
            "mtp_display_name": mtp.mtp_display_name if mtp else _MTP_DISPLAY_NAME,
            "mtp_slot_id": mtp.slot_id if mtp else None,
            "mtp_draft_model": mtp.draft.repo_id if mtp and mtp.draft else None,
            "mtp_identity": (
                f"{mtp.draft.repo_id}@{mtp.draft.revision}"
                if mtp and mtp.draft else None
            ),
            "global_apc_enabled": self._global_apc_enabled(slot),
            # Compatibility alias for callers that still treat the selected
            # slot as a DFlash boolean.  The generic acceleration mode remains
            # authoritative for new callers.
            "dflash_enabled": bool(dflash and slot.slot_id == dflash.slot_id),
        }

    def route_for_acceleration(self, slot: ModelSlot, mode: str) -> ModelSlot:
        """Resolve exactly one route for a target and enforce registry gates."""
        mode = str(mode or "").strip().lower()
        if mode not in _ACCELERATION_MODES:
            raise ModelSlotError(
                f"acceleration_mode must be one of {sorted(_ACCELERATION_MODES)}"
            )
        route = {
            "off": self.ar_variant(slot),
            "dflash": self.dflash_variant(slot),
            "mtp": self.mtp_variant(slot),
        }[mode]
        if route is None:
            raise ModelSlotError(
                f"no {mode} route is configured for target {slot.model!r}"
            )
        if not route.selectable:
            raise ModelSlotError(
                f"{mode} route is not selectable until verified: {route.slot_id!r}"
            )
        return route


def _canonical_slot_data(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in (
        "slot_id", "label", "selectable", "kind", "target", "draft", "routing", "runtime"
    )}


def slot_digest(raw: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonical_slot_data(raw), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelSlotError(f"{field} must be a non-empty string")
    return value


def _require_revision(value: Any, field: str) -> str:
    value = _require_text(value, field).lower()
    if len(value) != 40 or any(char not in _HEX40 for char in value):
        raise ModelSlotError(f"{field} must be an immutable 40-character revision")
    return value


def _parse_slot(raw: Any) -> ModelSlot:
    if not isinstance(raw, dict):
        raise ModelSlotError("each slot must be an object")
    slot_id = _require_text(raw.get("slot_id"), "slot_id")
    label = _require_text(raw.get("label"), f"{slot_id}.label")
    if not isinstance(raw.get("selectable"), bool):
        raise ModelSlotError(f"{slot_id}.selectable must be boolean")
    kind = raw.get("kind")
    if kind not in {"target_only", "target_plus_draft"}:
        raise ModelSlotError(f"{slot_id}.kind is invalid")
    target_raw = raw.get("target")
    if not isinstance(target_raw, dict):
        raise ModelSlotError(f"{slot_id}.target must be an object")
    target = TargetSpec(_require_text(target_raw.get("repo_id"), f"{slot_id}.target.repo_id"),
                        _require_revision(target_raw.get("revision"), f"{slot_id}.target.revision"))
    draft_raw = raw.get("draft")
    draft = None
    if draft_raw is not None:
        if not isinstance(draft_raw, dict):
            raise ModelSlotError(f"{slot_id}.draft must be null or an object")
        engine = _require_text(draft_raw.get("engine"), f"{slot_id}.draft.engine")
        if engine not in SUPPORTED_ADAPTERS:
            raise ModelSlotError(f"{slot_id} uses unsupported adapter {engine!r}")
        load_bits_raw = draft_raw.get("load_bits")
        block_size_raw = draft_raw.get("block_size")
        if load_bits_raw is None:
            load_bits = None
        elif type(load_bits_raw) is int and load_bits_raw > 0:
            load_bits = load_bits_raw
        else:
            raise ModelSlotError(f"{slot_id}.draft.load_bits must be null or positive integer")
        if type(block_size_raw) is not int or block_size_raw <= 0:
            raise ModelSlotError(f"{slot_id}.draft.block_size must be a positive integer")
        block_size = block_size_raw
        draft = DraftSpec(engine, _require_text(draft_raw.get("repo_id"), f"{slot_id}.draft.repo_id"),
                          _require_revision(draft_raw.get("revision"), f"{slot_id}.draft.revision"),
                          load_bits, block_size)
    if (kind == "target_only") != (draft is None):
        raise ModelSlotError(f"{slot_id}.kind does not match draft")
    routing = raw.get("routing")
    runtime = raw.get("runtime")
    if not isinstance(routing, dict) or not isinstance(runtime, dict):
        raise ModelSlotError(f"{slot_id}.routing/runtime must be objects")
    penalty = runtime.get("repetition_penalty")
    if penalty is not None and (
        type(penalty) not in (int, float) or not 1.0 <= penalty <= 2.0
    ):
        raise ModelSlotError(f"{slot_id}.runtime.repetition_penalty must be a number in 1.0..2.0")
    min_block = runtime.get("draft_min_block_size")
    if min_block is not None and (
        draft is None or draft.engine == _MTP_ADAPTER
        or type(min_block) is not int or not 2 <= min_block <= draft.block_size
    ):
        raise ModelSlotError(
            f"{slot_id}.runtime.draft_min_block_size must be a DFlash integer in 2..draft.block_size"
        )
    if routing.get("text") not in {"native_ar", "speculative"} or routing.get("vision") not in {"native_ar", "speculative"}:
        raise ModelSlotError(f"{slot_id}.routing is invalid")
    if draft is None and (routing["text"] != "native_ar" or routing["vision"] != "native_ar"):
        raise ModelSlotError(f"{slot_id} target-only routing must be native_ar")
    # mlx-vlm 0.7.x serves DFlash text and vision natively with its own APC, so
    # DFlash slots declare the same exact-APC contract as target-only slots.
    if runtime.get("apc") == "native" and (draft is None or draft.engine != _MTP_ADAPTER):
        if runtime.get("apc_enabled") is not True:
            raise ModelSlotError(
                f"{slot_id} native AR requires runtime.apc_enabled=true"
            )
        entries = runtime.get("apc_exact_cache_entries")
        if entries != 2:
            raise ModelSlotError(
                f"{slot_id} native AR requires runtime.apc_exact_cache_entries=2"
            )
        memory_gb = runtime.get("apc_memory_max_gb")
        if memory_gb is not None and (type(memory_gb) not in (int, float) or memory_gb <= 0):
            raise ModelSlotError(f"{slot_id}.runtime.apc_memory_max_gb must be a positive number")
        template_policy = runtime.get("apc_template_policy")
        if template_policy not in _NATIVE_APC_TEMPLATE_POLICIES:
            raise ModelSlotError(
                f"{slot_id}.runtime.apc_template_policy must be one of "
                f"{sorted(_NATIVE_APC_TEMPLATE_POLICIES)!r}"
            )
    if draft is not None and draft.engine == _MTP_ADAPTER:
        if runtime.get("draft_kind") != "mtp":
            raise ModelSlotError(f"{slot_id} MTP draft requires runtime draft_kind='mtp'")
        if runtime.get("rollback_mode") != "tape_replay":
            raise ModelSlotError(
                f"{slot_id} MTP draft requires runtime rollback_mode='tape_replay'"
            )
        weight_file = runtime.get("mtp_weight_file")
        weight_sha256 = runtime.get("mtp_weight_sha256")
        if not isinstance(weight_file, str) or not weight_file.endswith(".safetensors"):
            raise ModelSlotError(f"{slot_id}.runtime.mtp_weight_file is invalid")
        if (
            not isinstance(weight_sha256, str)
            or len(weight_sha256) != 64
            or any(char not in set("0123456789abcdef") for char in weight_sha256.lower())
        ):
            raise ModelSlotError(f"{slot_id}.runtime.mtp_weight_sha256 is invalid")
        if type(runtime.get("mtp_weight_size_bytes")) is not int or runtime["mtp_weight_size_bytes"] <= 0:
            raise ModelSlotError(f"{slot_id}.runtime.mtp_weight_size_bytes is invalid")
        adaptive_depth = runtime.get("mtp_adaptive_depth", False)
        if type(adaptive_depth) is not bool:
            raise ModelSlotError(f"{slot_id}.runtime.mtp_adaptive_depth must be boolean")
        max_block_size = runtime.get("mtp_max_block_size", draft.block_size)
        if type(max_block_size) is not int or not draft.block_size <= max_block_size <= 8:
            raise ModelSlotError(
                f"{slot_id}.runtime.mtp_max_block_size must be an integer in "
                f"{draft.block_size}..8"
            )
        if not adaptive_depth and max_block_size != draft.block_size:
            raise ModelSlotError(
                f"{slot_id}.runtime.mtp_max_block_size requires mtp_adaptive_depth=true"
            )
    digest = slot_digest(raw)
    return ModelSlot(slot_id, label, raw["selectable"], kind, target, draft, dict(routing), dict(runtime), digest)


def validate_registry(payload: Any) -> ModelSlotRegistry:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ModelSlotError("model-slot registry must use schema_version=2")
    default = _require_text(payload.get("default_slot_id"), "default_slot_id")
    legacy = payload.get("legacy_model_defaults")
    raw_slots = payload.get("slots")
    if not isinstance(legacy, dict) or not isinstance(raw_slots, list) or not raw_slots:
        raise ModelSlotError("registry requires legacy_model_defaults and slots")
    slots = tuple(_parse_slot(raw) for raw in raw_slots)
    ids = [slot.slot_id for slot in slots]
    if len(ids) != len(set(ids)):
        raise ModelSlotError("slot_id values must be unique")
    by_id = {slot.slot_id: slot for slot in slots}
    if default not in by_id:
        raise ModelSlotError("default_slot_id is unknown")
    pairs: set[tuple[str, str, str]] = set()
    for slot in slots:
        pair = (slot.target.repo_id, slot.target.revision,
                "" if slot.draft is None else f"{slot.draft.repo_id}@{slot.draft.revision}")
        if pair in pairs:
            raise ModelSlotError(f"ambiguous duplicate target/draft pair: {pair}")
        pairs.add(pair)
    for model, mapped_id in legacy.items():
        _require_text(model, "legacy model")
        if mapped_id not in by_id or by_id[mapped_id].model != model:
            raise ModelSlotError(f"invalid legacy mapping for {model!r}")
    return ModelSlotRegistry(SCHEMA_VERSION, default, dict(legacy), slots)


def load_registry(path: str | os.PathLike[str] | None = None) -> ModelSlotRegistry:
    registry_path = Path(path) if path is not None else REGISTRY_PATH
    try:
        with registry_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSlotError(f"cannot load model-slot registry: {registry_path}") from exc
    return validate_registry(payload)


def _registry_mtime() -> int | None:
    try:
        return REGISTRY_PATH.stat().st_mtime_ns
    except OSError:
        return None


REGISTRY = load_registry()
DEFAULT_SLOT = REGISTRY.get(REGISTRY.default_slot_id)
_REGISTRY_MTIME = _registry_mtime()
_REGISTRY_LOCK = threading.Lock()


def current_registry() -> ModelSlotRegistry:
    """Current registry; reloaded when model_slots.json changes on disk.

    Long-lived owners (agent_server's watchdog) used to keep the import-time
    registry, so after a runtime edit they relaunched the model server with a
    stale slot digest that the fresh launcher rejected, in a loop. An invalid
    edit keeps the last good registry.
    """
    global REGISTRY, DEFAULT_SLOT, _REGISTRY_MTIME
    mtime = _registry_mtime()
    if mtime == _REGISTRY_MTIME:
        return REGISTRY
    with _REGISTRY_LOCK:
        if mtime != _REGISTRY_MTIME:
            try:
                registry = load_registry()
                default = registry.get(registry.default_slot_id)
            except ModelSlotError:
                return REGISTRY
            REGISTRY, DEFAULT_SLOT, _REGISTRY_MTIME = registry, default, mtime
    return REGISTRY


def get_slot(slot_id: str) -> ModelSlot:
    return current_registry().get(slot_id)


def get_slot_for_model(model: str) -> ModelSlot:
    return current_registry().for_model(model)


def resolve_slot(*, slot_id: str | None = None, model: str | None = None) -> ModelSlot:
    return current_registry().resolve(slot_id=slot_id, model=model)


def dflash_metadata(slot: ModelSlot) -> dict[str, Any]:
    return current_registry().dflash_metadata(slot)


def route_for_dflash(slot: ModelSlot, enabled: bool) -> ModelSlot:
    return current_registry().route_for_dflash(slot, enabled)


def acceleration_metadata(slot: ModelSlot) -> dict[str, Any]:
    return current_registry().acceleration_metadata(slot)


def route_for_acceleration(slot: ModelSlot, mode: str) -> ModelSlot:
    return current_registry().route_for_acceleration(slot, mode)


def _default_hf_cache_root() -> Path:
    configured = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    configured = os.environ.get("HF_HOME")
    if configured:
        return Path(configured).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "huggingface" / "hub"


def _repo_cache_dir(cache_root: Path, repo_id: str) -> Path:
    # This is the cache naming contract used by Hugging Face Hub for ordinary
    # model repositories.  The registry still stores repo IDs, never paths.
    return cache_root / f"models--{repo_id.replace('/', '--')}"


def _snapshot_for(
    cache_root: Path, repo_id: str, revision: str, *, role: str, slot_id: str,
) -> Path:
    repo_dir = _repo_cache_dir(cache_root, repo_id)
    snapshot = repo_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise ModelSlotError(
            f"{role} snapshot missing for {slot_id}: {repo_id}@{revision}"
        )
    main_ref = repo_dir / "refs" / "main"
    # ``hf download --revision <commit>`` may materialize the exact commit
    # snapshot without creating ``refs/main``. The commit-named snapshot is
    # authoritative for an explicitly pinned revision; when a main ref is
    # present, still require it to agree and fail closed on a mismatch.
    if main_ref.exists() or main_ref.is_symlink():
        if not main_ref.is_file():
            raise ModelSlotError(f"{role} main ref unreadable for {slot_id}: {main_ref}")
        try:
            referenced = main_ref.read_text(encoding="utf-8").strip().lower()
        except OSError:
            raise ModelSlotError(f"{role} main ref unreadable for {slot_id}: {main_ref}") from None
        if referenced != revision:
            raise ModelSlotError(
                f"{role} revision is not the pinned main ref for {slot_id}: {referenced!r}"
            )
    incomplete = list(snapshot.rglob("*.incomplete"))
    if incomplete:
        raise ModelSlotError(
            f"{role} snapshot contains incomplete payload: {incomplete[0].name}"
        )
    return snapshot


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ModelSlotError(f"{role} has invalid JSON: {path}") from None
    if not isinstance(payload, dict):
        raise ModelSlotError(f"{role} must be a JSON object: {path}")
    return payload


def _require_snapshot_file(snapshot: Path, filename: str, *, role: str) -> Path:
    path = snapshot / filename
    if not path.is_file():
        raise ModelSlotError(f"{role} snapshot missing {filename}: {snapshot}")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ModelSlotError(f"{role} snapshot has an unreadable {filename}: {snapshot}") from None
    if resolved.name.endswith(".incomplete"):
        raise ModelSlotError(f"{role} snapshot points {filename} at incomplete payload")
    return path


def _required_weight_files(snapshot: Path) -> set[str]:
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
        except (OSError, json.JSONDecodeError, AttributeError):
            raise ModelSlotError(f"invalid weight index: {index}") from None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelSlotError(f"weight index has no weight_map: {index}")
        names = {str(name) for name in weight_map.values()}
        if any(
            not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not name.endswith(".safetensors")
            for name in names
        ):
            raise ModelSlotError(f"weight index contains an unsafe filename: {index}")
        return names
    weights = {path.name for path in snapshot.glob("*.safetensors")}
    if not weights:
        raise ModelSlotError(f"no safetensors weights in snapshot: {snapshot}")
    return weights


def preflight_target(slot: ModelSlot, *, cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Verify a pinned local target snapshot before lifecycle ownership changes."""
    root = Path(cache_root).expanduser() if cache_root is not None else _default_hf_cache_root()
    snapshot = _snapshot_for(
        root, slot.target.repo_id, slot.target.revision,
        role="target", slot_id=slot.slot_id,
    )
    _require_snapshot_file(snapshot, "config.json", role="target")
    _require_snapshot_file(snapshot, "tokenizer_config.json", role="target")
    _require_snapshot_file(snapshot, "tokenizer.json", role="target")
    if slot.target.repo_id == "mlx-community/Qwen3.8-27B-4bit":
        if not any(
            (snapshot / filename).is_file()
            for filename in ("processor_config.json", "preprocessor_config.json")
        ):
            raise ModelSlotError(f"target snapshot missing processor metadata: {snapshot}")
    required = _required_weight_files(snapshot)
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    if missing:
        raise ModelSlotError(f"target snapshot missing indexed weights: {missing[0]}")
    for name in sorted(required):
        _require_snapshot_file(snapshot, name, role="target")
    metadata = _read_json(snapshot / "config.json", role="target config.json")
    if slot.target.repo_id == "mlx-community/Qwen3.8-27B-4bit":
        text_config = metadata.get("text_config")
        architectures = metadata.get("architectures")
        quantization = metadata.get("quantization_config")
        if (
            not isinstance(text_config, dict)
            or not isinstance(architectures, list)
            or not isinstance(quantization, dict)
            or metadata.get("model_type") != "qwen3_5"
            or "Qwen3_5ForConditionalGeneration" not in architectures
            or type(text_config.get("hidden_size")) is not int
            or text_config.get("hidden_size") != 5120
            or type(text_config.get("vocab_size")) is not int
            or text_config.get("vocab_size") != 248320
            or type(text_config.get("num_hidden_layers")) is not int
            or text_config.get("num_hidden_layers") != 64
            or type(quantization.get("bits")) is not int
            or quantization.get("bits") != 4
            or quantization.get("mode") != "affine"
        ):
            raise ModelSlotError("qwen38-27b-ar target metadata does not match the pinned AR contract")
    return snapshot


def _safetensor_keys(snapshot: Path, required: set[str], *, role: str) -> set[str]:
    files = sorted(snapshot / name for name in required)
    if not files:
        raise ModelSlotError(f"{role} snapshot has no safetensors payload: {snapshot}")
    try:
        from safetensors import safe_open
    except ImportError:
        raise ModelSlotError(
            f"safetensors is required for strict {role} tensor preflight"
        ) from None
    keys: set[str] = set()
    try:
        for path in files:
            with safe_open(str(path), framework="np", device="cpu") as handle:
                keys.update(handle.keys())
    except Exception as exc:
        raise ModelSlotError(f"{role} safetensors header is unreadable: {exc}") from None
    return keys


_MTP_WEIGHT_SUFFIXES = frozenset({
    "fc.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "norm.weight",
})
_MTP_WEIGHT_PREFIXES = ("language_model.mtp.", "mtp.")


def _safetensor_header(path: Path, *, role: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read tensor names/shapes/dtypes from a safetensors header only."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ModelSlotError(
            f"safetensors is required for strict {role} tensor preflight"
        ) from None
    header: dict[str, tuple[tuple[int, ...], str]] = {}
    try:
        with safe_open(str(path), framework="np", device="cpu") as handle:
            for key in handle.keys():
                view = handle.get_slice(key)
                header[str(key)] = (
                    tuple(int(value) for value in view.get_shape()),
                    str(view.get_dtype()),
                )
    except Exception as exc:
        raise ModelSlotError(f"{role} safetensors header is unreadable: {exc}") from None
    return header


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight_mtp_draft(
    slot: ModelSlot,
    snapshot: Path,
    metadata: dict[str, Any],
    target_snapshot: Path,
) -> Path:
    """Strictly verify a Qwen MTP sidecar without loading its arrays."""
    draft = slot.draft
    assert draft is not None
    runtime = slot.runtime
    weight_file = runtime.get("mtp_weight_file")
    expected_sha = str(runtime.get("mtp_weight_sha256") or "").lower()
    expected_size = runtime.get("mtp_weight_size_bytes")
    if not isinstance(weight_file, str) or not weight_file.endswith(".safetensors"):
        raise ModelSlotError(f"MTP weight filename is invalid: {snapshot}")
    weight_path = _require_snapshot_file(snapshot, weight_file, role="MTP draft")
    resolved_weight = weight_path.resolve(strict=True)
    if resolved_weight.name.endswith(".incomplete"):
        raise ModelSlotError("MTP draft points at an incomplete weight payload")
    if type(expected_size) is not int or resolved_weight.stat().st_size != expected_size:
        raise ModelSlotError(
            f"MTP draft weight size mismatch: expected {expected_size}, "
            f"got {resolved_weight.stat().st_size}"
        )
    # A normal HF snapshot points at a content-addressed blob whose basename is
    # the expected hash.  Verify the bytes as well when a nonstandard local
    # materialization does not preserve that cache invariant.
    if resolved_weight.name.lower() != expected_sha:
        if _sha256_file(resolved_weight) != expected_sha:
            raise ModelSlotError("MTP draft weight SHA-256 does not match the pinned sidecar")

    architectures = metadata.get("architectures")
    if (
        not isinstance(architectures, list)
        or not all(isinstance(item, str) for item in architectures)
        or "Qwen3_5ForConditionalGeneration" not in architectures
        or metadata.get("model_type") != "qwen3_5"
    ):
        raise ModelSlotError(f"MTP sidecar source config is not the Qwen3.8 target family: {snapshot}")

    target_metadata = _read_json(target_snapshot / "config.json", role="target config.json")
    target_text = target_metadata.get("text_config") or target_metadata
    source_text = metadata.get("text_config") or metadata
    if not isinstance(target_text, dict) or not isinstance(source_text, dict):
        raise ModelSlotError(f"MTP/target text_config metadata is invalid: {snapshot}")

    for field in (
        "hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "intermediate_size",
    ):
        if source_text.get(field) != target_text.get(field):
            raise ModelSlotError(
                f"MTP/target {field} mismatch: {source_text.get(field)!r} != "
                f"{target_text.get(field)!r}"
            )
    mtp_layers = source_text.get("mtp_num_hidden_layers")
    expected_mtp_layers = runtime.get("mtp_num_hidden_layers", 1)
    if type(mtp_layers) is not int or mtp_layers <= 0 or mtp_layers != expected_mtp_layers:
        raise ModelSlotError(f"MTP sidecar hidden-layer count is invalid: {mtp_layers!r}")
    if source_text.get("mtp_use_dedicated_embeddings") is not False:
        raise ModelSlotError("MTP sidecar requires shared target embeddings")
    source_quant = metadata.get("quantization_config") or source_text.get("quantization_config")
    target_quant = target_metadata.get("quantization_config") or target_text.get("quantization_config")
    if not isinstance(source_quant, dict) or not isinstance(target_quant, dict):
        raise ModelSlotError("MTP/target quantization metadata is missing")
    for field in ("group_size", "bits", "mode"):
        if source_quant.get(field) != target_quant.get(field):
            raise ModelSlotError(
                f"MTP/target quantization {field} mismatch: "
                f"{source_quant.get(field)!r} != {target_quant.get(field)!r}"
            )
    if source_quant.get("bits") != 4 or source_quant.get("mode") != "affine":
        raise ModelSlotError("MTP sidecar source must preserve the target 4-bit affine contract")

    index_path = _require_snapshot_file(snapshot, "model.safetensors.index.json", role="MTP draft")
    index = _read_json(index_path, role="MTP draft index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelSlotError(f"MTP draft index has no weight_map: {index_path}")
    mtp_entries: dict[str, str] = {}
    prefixes: set[str] = set()
    for key, filename in weight_map.items():
        if not isinstance(key, str):
            continue
        for prefix in _MTP_WEIGHT_PREFIXES:
            if key.startswith(prefix):
                prefixes.add(prefix)
                mtp_entries[key[len(prefix):]] = str(filename)
                break
    if prefixes != {"language_model.mtp."}:
        raise ModelSlotError(
            f"MTP draft index must contain one language_model.mtp.* key namespace: {index_path}"
        )
    if set(mtp_entries) != set(_MTP_WEIGHT_SUFFIXES):
        missing = sorted(set(_MTP_WEIGHT_SUFFIXES) - set(mtp_entries))
        extra = sorted(set(mtp_entries) - set(_MTP_WEIGHT_SUFFIXES))
        detail = f"missing {missing[0]}" if missing else f"unexpected {extra[0]}"
        raise ModelSlotError(f"MTP draft tensor contract mismatch: {detail}")
    if any(filename != weight_file for filename in mtp_entries.values()):
        raise ModelSlotError("MTP draft index maps required tensors outside the pinned sidecar")

    # The sidecar repository republishes the target index and appends the 15
    # MTP tensors.  Its base entries intentionally point at the target's
    # existing shards; requiring every index value to be ``model-mtp.safetensors``
    # would make a complete sidecar impossible without duplicating ~32 GB of
    # target weights.  Require an exact base-key/file-map match instead so a
    # mismatched target cannot be silently combined with this sidecar.
    target_index_path = _require_snapshot_file(
        target_snapshot, "model.safetensors.index.json", role="target"
    )
    target_index = _read_json(target_index_path, role="target weight index")
    target_weight_map = target_index.get("weight_map")
    if not isinstance(target_weight_map, dict) or not target_weight_map:
        raise ModelSlotError(f"target weight index has no weight_map: {target_index_path}")
    base_weight_map = {
        str(key): str(filename)
        for key, filename in weight_map.items()
        if isinstance(key, str) and not key.startswith(tuple(_MTP_WEIGHT_PREFIXES))
    }
    normalized_target_map = {
        str(key): str(filename)
        for key, filename in target_weight_map.items()
        if isinstance(key, str)
    }
    if base_weight_map != normalized_target_map:
        raise ModelSlotError(
            "MTP draft base tensor index does not exactly match the pinned target"
        )

    header = _safetensor_header(weight_path, role="MTP draft")
    expected_raw_keys = {f"language_model.mtp.{suffix}" for suffix in _MTP_WEIGHT_SUFFIXES}
    if set(header) != expected_raw_keys:
        missing = sorted(expected_raw_keys - set(header))
        extra = sorted(set(header) - expected_raw_keys)
        detail = f"missing {missing[0]}" if missing else f"unexpected {extra[0]}"
        raise ModelSlotError(f"MTP sidecar header contract mismatch: {detail}")

    hidden = int(target_text["hidden_size"])
    heads = int(target_text["num_attention_heads"])
    kv_heads = int(target_text["num_key_value_heads"])
    head_dim = int(target_text["head_dim"])
    intermediate = int(target_text["intermediate_size"])
    expected_shapes = {
        "fc.weight": (hidden, 2 * hidden),
        "pre_fc_norm_embedding.weight": (hidden,),
        "pre_fc_norm_hidden.weight": (hidden,),
        "layers.0.input_layernorm.weight": (hidden,),
        "layers.0.post_attention_layernorm.weight": (hidden,),
        # Qwen3.5/Qwen3.8 applies Q/K RMSNorm per attention head.  These
        # vectors are therefore head_dim-wide, not hidden_size-wide (Q) or
        # num_key_value_heads * head_dim-wide (K).
        "layers.0.self_attn.q_norm.weight": (head_dim,),
        "layers.0.self_attn.k_norm.weight": (head_dim,),
        # Qwen3.5's query projection contains query plus the attention gate.
        "layers.0.self_attn.q_proj.weight": (2 * heads * head_dim, hidden),
        "layers.0.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
        "layers.0.self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
        "layers.0.self_attn.o_proj.weight": (hidden, heads * head_dim),
        "layers.0.mlp.gate_proj.weight": (intermediate, hidden),
        "layers.0.mlp.up_proj.weight": (intermediate, hidden),
        "layers.0.mlp.down_proj.weight": (hidden, intermediate),
        "norm.weight": (hidden,),
    }
    for suffix, expected_shape in expected_shapes.items():
        shape, dtype = header[f"language_model.mtp.{suffix}"]
        if shape != expected_shape:
            raise ModelSlotError(
                f"MTP tensor shape mismatch for {suffix}: {shape!r} != {expected_shape!r}"
            )
        if dtype.strip().upper() not in {"BF16", "BFLOAT16"}:
            raise ModelSlotError(f"MTP tensor {suffix} must be BF16, got {dtype!r}")
    return snapshot


def preflight_draft(slot: ModelSlot, *, cache_root: str | os.PathLike[str] | None = None,
                    target_snapshot: Path | None = None) -> Path:
    """Verify a pinned draft snapshot and its target/draft contract without loading weights."""
    if slot.draft is None:
        raise ModelSlotError(f"slot has no draft: {slot.slot_id}")
    root = Path(cache_root).expanduser() if cache_root is not None else _default_hf_cache_root()
    if target_snapshot is None:
        target_snapshot = preflight_target(slot, cache_root=root)
    draft = slot.draft
    snapshot = _snapshot_for(
        root, draft.repo_id, draft.revision,
        role="draft", slot_id=slot.slot_id,
    )
    _require_snapshot_file(snapshot, "config.json", role="draft")
    metadata = _read_json(snapshot / "config.json", role="draft config.json")
    if draft.engine == _MTP_ADAPTER:
        return _preflight_mtp_draft(slot, snapshot, metadata, target_snapshot)

    required = _required_weight_files(snapshot)
    for filename in sorted(required):
        _require_snapshot_file(snapshot, filename, role="draft")
    architectures = metadata.get("architectures")
    if not isinstance(architectures, list) or not all(isinstance(item, str) for item in architectures):
        raise ModelSlotError(f"draft architectures metadata is invalid: {snapshot}")

    target_metadata = _read_json(target_snapshot / "config.json", role="target config.json")
    target_text = target_metadata.get("text_config") or target_metadata
    if not isinstance(target_text, dict):
        raise ModelSlotError(f"target text_config metadata is invalid: {target_snapshot}")
    target_hidden = target_text.get("hidden_size")
    target_vocab = target_text.get("vocab_size")
    target_layers = target_text.get("num_hidden_layers")
    draft_hidden = metadata.get("hidden_size")
    draft_vocab = metadata.get("vocab_size")
    draft_layers = metadata.get("num_hidden_layers")
    draft_target_layers = metadata.get("num_target_layers")
    if any(
        type(value) is not int or value <= 0
        for value in (
            target_hidden, target_vocab, target_layers,
            draft_hidden, draft_vocab, draft_layers, draft_target_layers,
        )
    ):
        raise ModelSlotError(f"draft/target dimension metadata is invalid: {snapshot}")
    for field in ("hidden_size", "vocab_size"):
        if metadata.get(field) != target_text.get(field):
            raise ModelSlotError(
                f"draft/target {field} mismatch: {metadata.get(field)!r} != {target_text.get(field)!r}"
            )
    if draft_target_layers != target_layers:
        raise ModelSlotError(
            f"draft/target layer-count mismatch: {draft_target_layers!r} != {target_layers!r}"
        )
    dflash = metadata.get("dflash_config")
    if not isinstance(dflash, dict):
        raise ModelSlotError(f"draft is missing dflash_config: {snapshot}")
    target_layer_ids = dflash.get("target_layer_ids")
    if (
        not isinstance(target_layer_ids, list)
        or not target_layer_ids
        or any(type(item) is not int for item in target_layer_ids)
        or len(set(target_layer_ids)) != len(target_layer_ids)
        or not isinstance(target_layers, int)
        or any(item < 0 or item >= target_layers for item in target_layer_ids)
    ):
        raise ModelSlotError(f"draft target_layer_ids are invalid: {snapshot}")

    # z-lab/dflash#156: quantized MLX verifies at width 5 or below; a
    # checkpoint's native block (8, 16) is a ceiling for BF16 targets only.
    quantized = target_metadata.get("quantization") or target_metadata.get("quantization_config")
    if quantized and draft.block_size > _DFLASH_QUANTIZED_MAX_BLOCK:
        raise ModelSlotError(
            f"DFlash block_size must be {_DFLASH_QUANTIZED_MAX_BLOCK} or lower for a quantized target"
        )

    if draft.engine == "mlx_vlm_dflash2":
        if "DFlash2DraftModel" not in architectures:
            raise ModelSlotError("DFlash2 slot requires architecture DFlash2DraftModel")
        if slot.slot_id == "qwen38-27b-dflash2" and target_layer_ids != [5, 19, 33, 47, 61]:
            raise ModelSlotError("pinned Qwen3.8 DFlash2 target_layer_ids are invalid")
        if draft.load_bits is not None:
            raise ModelSlotError("mlx-vlm loads the DFlash2 draft unquantized; load_bits must be null")
        dtype = str(metadata.get("dtype") or metadata.get("torch_dtype") or "").lower()
        if dtype not in {"bfloat16", "bf16"}:
            raise ModelSlotError(f"DFlash2 draft must be BF16 on disk, got {dtype!r}")
        # A pre-quantized MLX export keeps BF16 scales; only affine 4/8-bit loads natively.
        draft_quant = metadata.get("quantization")
        if draft_quant is not None and (
            not isinstance(draft_quant, dict)
            or draft_quant.get("mode", "affine") != "affine"
            or draft_quant.get("bits") not in {4, 8}
            or type(draft_quant.get("group_size")) is not int
        ):
            raise ModelSlotError("DFlash2 draft quantization must be affine 4- or 8-bit")
        checkpoint_block_size = dflash.get("block_size", metadata.get("block_size"))
        if type(checkpoint_block_size) is not int or checkpoint_block_size <= 0:
            raise ModelSlotError("DFlash2 draft block_size metadata is invalid")
        if draft.block_size > checkpoint_block_size:
            raise ModelSlotError("DFlash2 block_size must not exceed the checkpoint block_size")
        for field in (
            "conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k",
        ):
            if type(dflash.get(field)) is not int or dflash[field] <= 0:
                raise ModelSlotError(f"DFlash2 draft metadata missing positive {field}")
        required_keys = {
            "candidate_selector.hidden_projection.weight",
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
            "fc.weight", "hidden_norm.weight", "norm.weight",
        }
        layer_suffixes = (
            "attention_conv.base_kernel",
            "attention_conv.kernel_projection.weight",
            "mlp_conv.base_kernel",
            "mlp_conv.kernel_projection.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
        required_keys.update(
            f"layers.{layer}.{suffix}"
            for layer in range(draft_layers)
            for suffix in layer_suffixes
        )
        present_keys = _safetensor_keys(snapshot, required, role="DFlash2 draft")
        # mlx-vlm's DFlash2 sanitize writes MLX exports' codebooks as "<name>.weight".
        present_keys |= {key[: -len(".weight")] for key in present_keys if key.endswith("_codebook.weight")}
        missing_keys = sorted(required_keys - present_keys)
        if missing_keys:
            raise ModelSlotError(f"DFlash2 draft missing required tensor: {missing_keys[0]}")
    elif draft.engine == "mlx_vlm_dflash_v1":
        if "DFlashDraftModel" not in architectures:
            raise ModelSlotError("DFlash v1 slot requires architecture DFlashDraftModel")
    else:
        raise ModelSlotError(f"unsupported draft adapter: {draft.engine!r}")
    return snapshot


def preflight_slot(slot: ModelSlot, *, cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Verify every local artifact required by a slot before lifecycle changes."""
    root = Path(cache_root).expanduser() if cache_root is not None else _default_hf_cache_root()
    target = preflight_target(slot, cache_root=root)
    if slot.draft is not None:
        preflight_draft(slot, cache_root=root, target_snapshot=target)
    return target
