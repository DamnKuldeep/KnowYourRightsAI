"""Probe the machine, then decide what to load.

The point of this module is that nothing downstream guesses. We read free VRAM and available
RAM at startup and pick a profile that leaves the configured headroom, so the app stays a
good citizen on a laptop that is also doing other work.

``torch`` is imported lazily — importing it costs seconds, and callers that only want the RAM
figures shouldn't pay that.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field

from .. import config

log = logging.getLogger(__name__)

MB = 1024 * 1024


@dataclass(frozen=True)
class ResourceSnapshot:
    cuda_available: bool
    device_name: str
    vram_total_mb: int
    vram_free_mb: int
    ram_total_mb: int
    ram_available_mb: int
    ram_percent: float
    process_rss_mb: int
    cpu_physical: int
    cpu_logical: int

    @property
    def vram_used_mb(self) -> int:
        return self.vram_total_mb - self.vram_free_mb

    def describe(self) -> str:
        if self.cuda_available:
            gpu = (f"{self.device_name}: {self.vram_free_mb} MiB free of {self.vram_total_mb} MiB "
                   f"({self.vram_used_mb} MiB already in use)")
        else:
            gpu = "no usable CUDA device"
        return (f"{gpu} | RAM {self.ram_available_mb} MB available of {self.ram_total_mb} MB "
                f"({self.ram_percent:.0f}% used) | CPU {self.cpu_physical}p/{self.cpu_logical}l")


@dataclass(frozen=True)
class ResourcePlan:
    """What we will actually load, and why."""

    profile: config.Profile
    snapshot: ResourceSnapshot
    embed_device: str          # "cuda" | "cpu"
    embed_dtype: str           # "float16" | "float32"
    rerank_backend: str        # "local" | "nim" | "none"
    rerank_model: str | None
    rerank_device: str
    rerank_dtype: str
    embed_batch: int
    rerank_batch: int
    ram_ok: bool               # enough RAM to survive the load spike?
    use_embedder: bool = True  # False in `lite`: BM25 only, no models at all
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def projected_free_vram_mb(self) -> int:
        if self.embed_device == "cpu":
            return self.snapshot.vram_free_mb
        return self.snapshot.vram_free_mb - self.profile.model_vram_mb

    def describe(self) -> str:
        lines = [f"profile      : {self.name}  ({self.profile.note})"]
        if self.use_embedder:
            lines.append(f"embedder     : {config.EMBED_MODEL} on "
                         f"{self.embed_device} / {self.embed_dtype}")
        else:
            lines.append("embedder     : none — BM25 keyword search only")
        if self.rerank_backend == "local":
            lines.append(f"reranker     : {self.rerank_model} on {self.rerank_device} / {self.rerank_dtype}")
        elif self.rerank_backend == "nim":
            lines.append(f"reranker     : NIM {config.NIM_RERANK_MODEL} (remote)")
        else:
            lines.append("reranker     : disabled — ranking falls back to fused RRF scores")
        if self.embed_device == "cuda":
            lines.append(f"VRAM         : ~{self.profile.model_vram_mb} MiB for models, "
                         f"~{self.projected_free_vram_mb} MiB left free "
                         f"(reserve {config.VRAM_RESERVE_MB} MiB)")
        lines.append(f"batches      : embed={self.embed_batch} rerank={self.rerank_batch}")
        if not self.ram_ok:
            lines.append(f"RAM WARNING  : {self.snapshot.ram_available_mb} MB available, "
                         f"{config.RAM_LOAD_HEADROOM_MB} MB wanted for the load spike — "
                         f"models load lazily and retrieval starts in FTS-only mode")
        for r in self.reasons:
            lines.append(f"  · {r}")
        return "\n".join(lines)


def _cpu_counts() -> tuple[int, int]:
    try:
        import psutil

        return psutil.cpu_count(logical=False) or 1, psutil.cpu_count() or 1
    except Exception:
        n = os.cpu_count() or 1
        return max(1, n // 2), n


def _ram() -> tuple[int, int, float, int]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss
        return vm.total // MB, vm.available // MB, float(vm.percent), rss // MB
    except Exception:
        return 0, 0, 0.0, 0


def _vram() -> tuple[bool, str, int, int]:
    """(cuda_available, device_name, total_mb, free_mb). Never raises."""
    try:
        import torch
    except Exception:
        return False, "", 0, 0
    try:
        if not torch.cuda.is_available():
            return False, "", 0, 0
        free, total = torch.cuda.mem_get_info()
        return True, torch.cuda.get_device_name(0), total // MB, free // MB
    except Exception as exc:  # a driver hiccup should degrade, not crash
        log.warning("CUDA probe failed, falling back to CPU: %s", exc)
        return False, "", 0, 0


def probe() -> ResourceSnapshot:
    """Measure the machine as it is right now."""
    cuda, name, vram_total, vram_free = _vram()
    ram_total, ram_avail, ram_pct, rss = _ram()
    phys, logical = _cpu_counts()
    return ResourceSnapshot(
        cuda_available=cuda,
        device_name=name,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        ram_total_mb=ram_total,
        ram_available_mb=ram_avail,
        ram_percent=ram_pct,
        process_rss_mb=rss,
        cpu_physical=phys,
        cpu_logical=logical,
    )


def _profile_by_name(name: str) -> config.Profile | None:
    if name == "lite":
        return config.LITE
    return next((p for p in config.PROFILES if p.name == name), None)


def _fits(profile: config.Profile, snap: ResourceSnapshot) -> bool:
    if not profile.use_embedder:
        return True                      # lite loads nothing, so it fits anywhere
    if profile.name in ("cpu", "cpu_lean"):
        # cpu_lean needs a key, since its reranking is remote.
        return profile.name == "cpu" or bool(config.NVIDIA_API_KEY)
    if not snap.cuda_available:
        return False
    return snap.vram_free_mb >= profile.model_vram_mb + config.VRAM_RESERVE_MB


def select_profile(snapshot: ResourceSnapshot | None = None,
                   requested: str | None = None) -> ResourcePlan:
    """Choose a profile against the probed machine.

    An explicit request always wins — if it doesn't fit we say so loudly rather than silently
    overriding the user, because someone forcing ``quality`` usually knows their machine is idle.
    """
    snap = snapshot or probe()
    want = (requested or config.PROFILE_REQUEST or "auto").strip().lower()
    reasons: list[str] = []

    if want != "auto":
        profile = _profile_by_name(want)
        if profile is None:
            reasons.append(f"unknown profile {want!r}; falling back to auto-selection")
            want = "auto"
        else:
            if not _fits(profile, snap):
                reasons.append(
                    f"forced profile {profile.name!r} wants "
                    f"{profile.model_vram_mb + config.VRAM_RESERVE_MB} MiB but only "
                    f"{snap.vram_free_mb} MiB is free — expect OOM pressure"
                )
            else:
                reasons.append(f"profile {profile.name!r} requested explicitly")
            return _build_plan(profile, snap, reasons)

    for profile in config.PROFILES:
        if _fits(profile, snap):
            if profile.name == "cpu" and snap.cuda_available:
                reasons.append("no GPU profile fits the free VRAM; running on CPU")
            else:
                reasons.append(
                    f"selected {profile.name!r}: needs {profile.model_vram_mb} MiB + "
                    f"{config.VRAM_RESERVE_MB} MiB reserve, {snap.vram_free_mb} MiB free"
                )
            return _build_plan(profile, snap, reasons)

    # Unreachable — the cpu profile always fits — but be explicit rather than clever.
    return _build_plan(config.PROFILES[-1], snap, reasons + ["no profile fit; forced cpu"])


def _build_plan(profile: config.Profile, snap: ResourceSnapshot,
                reasons: list[str]) -> ResourcePlan:
    on_gpu = profile.name not in ("cpu", "cpu_lean") and snap.cuda_available
    device = "cuda" if on_gpu else "cpu"
    # fp16 is a GPU optimisation; on CPU it is slower than fp32 for these models.
    dtype = "float16" if on_gpu else "float32"

    rerank_backend = profile.rerank_backend
    if rerank_backend == "nim" and not config.NVIDIA_API_KEY:
        rerank_backend = "none"
        reasons.append("NVIDIA_API_KEY is unset, so the NIM reranker is unavailable; "
                       "ranking will use fused RRF scores")

    ram_ok = snap.ram_available_mb >= config.RAM_LOAD_HEADROOM_MB
    if not ram_ok:
        reasons.append(
            f"only {snap.ram_available_mb} MB RAM available; loading bge-m3 peaks around "
            f"{config.RAM_LOAD_HEADROOM_MB} MB. Close something, or retrieval will run "
            f"keyword-only until memory frees up."
        )

    if not profile.use_embedder:
        reasons.append("lite profile: no models are loaded — retrieval is BM25 only "
                       "(Recall@5 95% against 100%, but it fits in under 1 GB)")
        ram_ok = True          # nothing large is being loaded, so the RAM floor does not apply

    return ResourcePlan(
        profile=profile,
        use_embedder=profile.use_embedder,
        snapshot=snap,
        embed_device=device,
        embed_dtype=dtype,
        rerank_backend=rerank_backend,
        rerank_model=profile.rerank_model if rerank_backend == "local" else None,
        rerank_device=device,
        rerank_dtype=dtype,
        embed_batch=profile.embed_batch,
        rerank_batch=profile.rerank_batch,
        ram_ok=ram_ok,
        reasons=tuple(reasons),
    )


# The selected plan is process-wide state: one model set, one GPU, one server.
_PLAN: ResourcePlan | None = None


def get_plan(refresh: bool = False) -> ResourcePlan:
    global _PLAN
    if _PLAN is None or refresh:
        _PLAN = select_profile()
        log.info("resource plan:\n%s", _PLAN.describe())
    return _PLAN


def live_usage() -> dict[str, int | float | str]:
    """Cheap current-usage numbers for /api/health and the UI footer."""
    snap = probe()
    return {
        "profile": _PLAN.name if _PLAN else "unselected",
        "vram_total_mb": snap.vram_total_mb,
        "vram_free_mb": snap.vram_free_mb,
        "vram_used_mb": snap.vram_used_mb,
        "ram_total_mb": snap.ram_total_mb,
        "ram_available_mb": snap.ram_available_mb,
        "ram_percent": round(snap.ram_percent, 1),
        "process_rss_mb": snap.process_rss_mb,
        "platform": platform.platform(terse=True),
    }
