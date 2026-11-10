"""Room on the GPU before an engine loads.

Engines load on demand and used to pile up until one failed for lack of memory, with an error
that didn't say so. Now, when the engine about to load doesn't fit in the free memory (its figure
is "gpu_gb" in engine_info.json, measured), idle engines are unloaded first, least recently used
first. Engines marked "keep loaded", busy ones and ones used a moment ago stay.

If it still doesn't fit, the load is tried anyway (the figures are peaks, and on Windows a model
can spill into shared memory); should it then fail, the note returned here says why.
"""
import time

MARGIN_GB = 0.5       # free memory wanted on top of the engine's own figure
JUST_USED_SECONDS = 5  # an engine used this recently is probably mid-reading, between sentences


def make_room(engine, need_gb: float, free_gb, others: list, kept: set, log=print) -> str | None:
    """engine is about to load and needs need_gb. free_gb() -> GB free on the GPU right now.
    others: the other loaded GPU engines. kept: names of engines never unloaded for room.
    -> None when it fits (after unloading if needed), else a sentence saying what was short."""
    free = free_gb()
    if need_gb + MARGIN_GB <= free:
        return None
    stayed = []
    for other in sorted(others, key=lambda e: e.last_used):
        if need_gb + MARGIN_GB <= free:
            break
        idle = time.monotonic() - other.last_used
        if other.name in kept:
            stayed.append(f"{other.label} is kept loaded")
        elif idle < JUST_USED_SECONDS:
            stayed.append(f"{other.label} is in use")
        else:
            log(f"[load] {engine.label}: needs about {need_gb:g} GB and {free:.1f} GB is free, "
                f"so {other.label} (idle {idle / 60:.0f} min) is unloaded", flush=True)
            if other.unload(wait=False):
                free = free_gb()
            else:
                stayed.append(f"{other.label} is in use")
    if need_gb + MARGIN_GB <= free:
        return None
    note = f"It needs about {need_gb:g} GB of GPU memory and {free:.1f} GB is free"
    return note + (f" ({'; '.join(stayed)})." if stayed else ".")
