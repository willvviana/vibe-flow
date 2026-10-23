"""Memory preflight: refuse to load a model that cannot fit, using measured (not assumed) requirements."""
import psutil

HEADROOM_GB = 2.0      # always leave this much for Windows and other applications
HARD_FLOOR = 0.6       # below this fraction of the model's peak, loading would thrash swap -> refuse


class ResourceError(RuntimeError):
    """Raised before any large allocation so the app stays stable."""


def required_gb(spec) -> float:
    """Peak RAM the model needs. Uses the measured peak when we have one, otherwise the (padded) estimate."""
    return spec.measured_peak_gb or spec.ram_gb * 1.25


def check(spec, available_gb: float, device: str = "cpu", already_loaded: bool = False):
    """Pure decision function (unit-testable). Returns (level, message): 'ok' | 'warn' | 'refuse'."""
    if device != "cpu" or already_loaded:
        return "ok", ""
    need = required_gb(spec)
    basis = "measured" if spec.measured_peak_gb else "estimated"
    if available_gb < need * HARD_FLOOR:
        return "refuse", (f"Not enough free memory for {spec.name}: it needs about {need:.1f} GB ({basis} peak) but only "
                          f"{available_gb:.1f} GB is free. Close other applications (browsers, editors) and try again.")
    if available_gb < need + HEADROOM_GB:
        return "warn", (f"Memory is tight: {spec.name} peaks around {need:.1f} GB ({basis}) with {available_gb:.1f} GB free. "
                        "It may run slowly while Windows pages memory to disk.")
    return "ok", ""


def preflight(spec, hw: dict, device: str, already_loaded: bool = False) -> list:
    """Raise ResourceError if the model cannot safely load; return warnings otherwise."""
    avail = psutil.virtual_memory().available / 2**30      # live value, not the one captured earlier
    level, msg = check(spec, avail, device, already_loaded)
    if level == "refuse":
        raise ResourceError(msg)
    return [msg] if msg else []
