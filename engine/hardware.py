"""Hardware / runtime detection and device selection."""
import os
import platform
import shutil
import subprocess


def _gpu_names():
    if platform.system() != "Windows":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=10).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def detect():
    import psutil
    vm = psutil.virtual_memory()
    info = {
        "os": platform.platform(),
        "cpu_threads": os.cpu_count(),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_available_gb": round(vm.available / 2**30, 1),
        "gpus": _gpu_names(),
        "cuda": False, "cuda_name": None, "vram_gb": None,
        "torch": None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(cuda=True, cuda_name=p.name, vram_gb=round(p.total_memory / 2**30, 1))
    except ImportError:
        pass
    info["disk_free_gb"] = round(shutil.disk_usage(os.getcwd()).free / 2**30, 1)
    info["device"] = "cuda" if info["cuda"] else "cpu"
    info["device_note"] = (
        "NVIDIA CUDA GPU available." if info["cuda"] else
        "Running on CPU. Non-CUDA GPUs (e.g. AMD on Windows) are not used because PyTorch has no supported backend for them here.")
    return info


def choose_device(hw: dict, spec) -> str:
    """CUDA only if available and VRAM is enough for the model in fp32; otherwise CPU."""
    if hw.get("cuda") and (hw.get("vram_gb") or 0) >= spec.ram_gb:
        return "cuda"
    return "cpu"


def resource_warnings(hw: dict, spec, device: str) -> list:
    w = []
    if device == "cuda" and (hw.get("vram_gb") or 0) < spec.ram_gb:
        w.append(f"GPU has {hw['vram_gb']} GB VRAM; {spec.name} needs about {spec.ram_gb} GB.")
    if device == "cpu":
        w.append("CPU generation is slow (expect several times longer than the audio length).")
    return w
