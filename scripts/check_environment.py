"""
Environment Verification Script - SIH 2026 AI Engine
=====================================================
Checks and reports:
- Python version and system architecture
- PyTorch installation and version
- CUDA availability and GPU device information
- OpenCV installation status
- Ultralytics YOLO installation status

Fails gracefully if any dependency is missing and provides clear installation guidance.
"""

import sys
import platform
import subprocess
import shutil

# Ensure UTF-8 output when possible on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Color formatting for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def status_badge(passed: bool, warning: bool = False) -> str:
    if passed:
        return f"{GREEN}[PASS]{RESET}"
    elif warning:
        return f"{YELLOW}[WARN]{RESET}"
    else:
        return f"{RED}[FAIL]{RESET}"



def print_banner():
    banner = f"""
{CYAN}{BOLD}=============================================================
  SIH 2026 AI ENGINE - ENVIRONMENT VERIFICATION
  Multi-Camera Traffic Surveillance & Tracking Engine
============================================================={RESET}
"""
    print(banner)


def check_python() -> bool:
    print(f"{BOLD}1. Python Environment:{RESET}")
    version = sys.version.split()[0]
    major, minor, micro = sys.version_info[:3]
    exe_path = sys.executable
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    print(f"   * Python Executable : {exe_path}")
    print(f"   * Operating System  : {os_info}")
    print(f"   * Detected Version  : {version}", end=" ")

    # Recommended Python 3.10 or 3.11 for PyTorch and CV stacks
    if major == 3 and 9 <= minor <= 12:
        print(f"{status_badge(True)} (Recommended version)")
        return True
    else:
        print(f"{status_badge(False, warning=True)} (Version {version} detected. Python 3.10 or 3.11 is recommended for maximum library compatibility)")
        return True


def check_system_gpu() -> dict:
    """Check hardware GPU via nvidia-smi independent of PyTorch."""
    gpu_info = {"has_nvidia_smi": False, "gpu_name": None, "driver_version": None}
    if shutil.which("nvidia-smi"):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            ).strip()
            if output:
                lines = output.splitlines()
                parts = lines[0].split(", ")
                gpu_info["has_nvidia_smi"] = True
                gpu_info["gpu_name"] = parts[0].strip()
                gpu_info["driver_version"] = parts[1].strip() if len(parts) > 1 else "Unknown"
        except Exception:
            pass
    return gpu_info


def check_pytorch(sys_gpu: dict) -> bool:
    print(f"\n{BOLD}2. PyTorch & GPU Acceleration:{RESET}")
    try:
        import torch
        torch_ver = torch.__version__
        print(f"   * PyTorch Installed : {status_badge(True)} (Version: {torch_ver})")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_ver = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

            print(f"   * CUDA Available    : {status_badge(True)} (CUDA {cuda_ver})")
            print(f"   * Active GPU Device : {gpu_name} (Count: {device_count})")
            print(f"   * Total VRAM        : {total_mem:.2f} GB")
            return True
        else:
            print(f"   * CUDA Available    : {status_badge(False, warning=True)} (Running in CPU mode)")
            if sys_gpu["has_nvidia_smi"]:
                print(f"   {YELLOW}[INFO]{RESET} Physical GPU detected ({sys_gpu['gpu_name']}), but current PyTorch build does not have CUDA enabled.")
                print(f"   {YELLOW}[ACTION]{RESET} Reinstall PyTorch with CUDA:")
                print(f"          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            return True
    except ImportError:
        print(f"   * PyTorch Installed : {status_badge(False)} (Not installed)")
        if sys_gpu["has_nvidia_smi"]:
            print(f"   {YELLOW}[INFO]{RESET} Hardware detected: {sys_gpu['gpu_name']} (Driver: {sys_gpu['driver_version']})")
            print(f"   {YELLOW}[ACTION]{RESET} Install PyTorch with CUDA support:")
            print(f"          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        else:
            print(f"   {YELLOW}[ACTION]{RESET} Install PyTorch using: pip install torch torchvision")
        return False


def check_opencv() -> bool:
    print(f"\n{BOLD}3. OpenCV (Computer Vision Library):{RESET}")
    try:
        import cv2
        print(f"   * OpenCV Installed  : {status_badge(True)} (Version: {cv2.__version__})")
        return True
    except ImportError:
        print(f"   * OpenCV Installed  : {status_badge(False)} (Not installed)")
        print(f"   {YELLOW}[ACTION]{RESET} Install OpenCV using: pip install opencv-python")
        return False


def check_ultralytics() -> bool:
    print(f"\n{BOLD}4. Ultralytics YOLO (Detection Framework):{RESET}")
    try:
        import ultralytics
        print(f"   * Ultralytics YOLO  : {status_badge(True)} (Version: {ultralytics.__version__})")
        return True
    except ImportError:
        print(f"   * Ultralytics YOLO  : {status_badge(False)} (Not installed)")
        print(f"   {YELLOW}[ACTION]{RESET} Install Ultralytics using: pip install ultralytics")
        return False


def main():
    print_banner()

    sys_gpu = check_system_gpu()
    py_ok = check_python()
    torch_ok = check_pytorch(sys_gpu)
    cv_ok = check_opencv()
    yolo_ok = check_ultralytics()

    all_ready = py_ok and torch_ok and cv_ok and yolo_ok

    print(f"\n{CYAN}{BOLD}============================================================={RESET}")
    print(f"{BOLD}PHASE 1 ENVIRONMENT STATUS:{RESET}")
    if all_ready:
        print(f"  {GREEN}{BOLD}[OK] ALL CORE PACKAGES INSTALLED AND READY FOR PHASE 2.{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}[!] SETUP IN PROGRESS: Some packages are not yet installed.{RESET}")
        print(f"  Follow the instructions in the README to activate your virtual environment")
        print(f"  and install the Phase 1 dependencies.")
    print(f"{CYAN}{BOLD}============================================================={RESET}\n")



if __name__ == "__main__":
    main()
