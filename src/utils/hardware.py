"""
Hardware and Compute Environment Utilities for Device Resolution and Profiling.
"""

import os
import platform
import psutil
from typing import Tuple, Optional, Any
import torch


def resolve_device(device: Optional[str] = None) -> Tuple[torch.device, str, str]:
    """
    Resolves compute device and hardware target description.
    Returns: (torch.device, device_str, hw_target_description)
    """
    if device is not None:
        dev_str = device
    elif torch.cuda.is_available():
        dev_str = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev_str = "mps"
    else:
        dev_str = "cpu"

    dev = torch.device(dev_str)

    if "cuda" in dev_str:
        hw_target = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    elif "mps" in dev_str:
        hw_target = f"Apple Silicon MPS ({platform.machine()})"
    else:
        hw_target = f"Host CPU ({platform.processor() or platform.machine()})"

    return dev, dev_str, hw_target


def get_process_memory_gb() -> float:
    """Returns current process RAM utilization in gigabytes."""
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss / (1024**3))


def release_model_vram(model: Any):
    """
    Safely releases CUDA VRAM allocated by a PyTorch model or Accelerate-dispatched pipeline.
    Strips Accelerate execution hooks, zeros out parameter and buffer CUDA storage,
    and returns memory blocks to the CUDA caching allocator.
    """
    if model is None:
        return

    # 1. Handle Hugging Face pipeline wrapping
    if hasattr(model, "model") and getattr(model, "model", None) is not None:
        release_model_vram(getattr(model, "model"))

    # 2. Detach Accelerate hooks recursively
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    except Exception:
        pass

    # 3. Recursively strip Accelerate hooks and CUDA storage from submodules
    if hasattr(model, "modules"):
        try:
            for submod in model.modules():
                if hasattr(submod, "_hf_hook"):
                    try:
                        delattr(submod, "_hf_hook")
                    except Exception:
                        pass
                for p_name, p in list(getattr(submod, "_parameters", {}).items()):
                    if p is not None and isinstance(p, torch.Tensor) and p.is_cuda:
                        p.grad = None
                        p.data = torch.empty(0, device="cpu")
                for b_name, b in list(getattr(submod, "_buffers", {}).items()):
                    if b is not None and isinstance(b, torch.Tensor) and b.is_cuda:
                        b.data = torch.empty(0, device="cpu")
        except Exception:
            pass

    # 4. Fallback for root module parameters and buffers
    try:
        if hasattr(model, "parameters"):
            for p in model.parameters():
                if p is not None and isinstance(p, torch.Tensor) and p.is_cuda:
                    p.grad = None
                    p.data = torch.empty(0, device="cpu")
        if hasattr(model, "buffers"):
            for b in model.buffers():
                if b is not None and isinstance(b, torch.Tensor) and b.is_cuda:
                    b.data = torch.empty(0, device="cpu")
    except Exception:
        pass

    # 5. Move module to CPU if possible (only after hooks are stripped)
    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass


def deep_cuda_purge():
    """
    Performs multi-cycle garbage collection and empties CUDA cache and IPC allocations across all devices.
    """
    import gc
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass


def inspect_gpu_processes():
    """
    Inspects running compute processes on all GPUs using nvidia-smi.
    Prints other PIDs occupying VRAM and alerts if zombie processes are detected.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return
        current_pid = os.getpid()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        other_processes = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    pid = int(parts[0])
                    pname = parts[1]
                    pmem = parts[2]
                    if pid != current_pid:
                        other_processes.append((pid, pname, pmem))
                except ValueError:
                    pass
        if other_processes:
            print("\n  [GPU Resource Alert] Other processes detected occupying GPU VRAM:")
            for pid, pname, pmem in other_processes:
                print(f"    - PID {pid} ({pname}): {pmem}")
            print("  If these are lingering zombie processes from previous interrupted runs,")
            print("  you can reclaim full VRAM by terminating them with: kill -9 <PID>")
            print("  Or run with '--clean_gpu' to automatically terminate stale benchmark processes.\n")
    except Exception:
        pass


def clean_stale_gpu_processes(auto_kill: bool = False) -> int:
    """
    Detects and optionally terminates stale or orphaned Python GPU processes occupying VRAM.
    Returns the count of terminated processes.
    """
    import subprocess
    import time
    import signal

    killed_count = 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return 0

        current_pid = os.getpid()
        my_parents = set()
        try:
            curr_proc = psutil.Process(current_pid)
            for p in curr_proc.parents():
                my_parents.add(p.pid)
        except Exception:
            pass

        lines = [line.strip() for line in out.splitlines() if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    pid = int(parts[0])
                    pname = parts[1]
                    pmem = parts[2]
                    if pid == current_pid or pid in my_parents:
                        continue

                    # Check process cmdline if accessible
                    is_candidate = False
                    cmd_str = ""
                    try:
                        p = psutil.Process(pid)
                        cmd_str = " ".join(p.cmdline()).lower()
                        if "python" in pname.lower() or "python" in cmd_str:
                            if any(k in cmd_str for k in ["run_all_benchmarks", "benchmark", "indobert", "torch", "accelerate", "train"]):
                                is_candidate = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                    if auto_kill and is_candidate:
                        print(f"  [GPU Process Cleanup] Terminating orphaned process PID {pid} ({pname}, occupying {pmem})...")
                        try:
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(0.5)
                            if psutil.pid_exists(pid):
                                os.kill(pid, signal.SIGKILL)
                            killed_count += 1
                        except Exception as kill_err:
                            print(f"  [GPU Process Cleanup] Failed to kill PID {pid}: {kill_err}")
                except ValueError:
                    pass

        if killed_count > 0:
            print(f"  [GPU Process Cleanup] Reclaimed VRAM by terminating {killed_count} stale process(es). Waiting 2s for GPU driver release...")
            time.sleep(2.0)
            deep_cuda_purge()
    except Exception:
        pass

    return killed_count




