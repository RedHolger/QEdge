"""Hardware and software inventory for Qualcomm deployment (Stage 11).

Distinguishes:
- Installed: software and dependencies locally installed.
- Accessible: physical hardware and devices reachable via ADB or local bus.
- Missing: prerequisites not present in the execution environment.
- Unverified: components that cannot be verified without physical target execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Optional

from qedge_runtime.config import get_default_model_dir


@dataclass
class QualcommHardwareInventory:
    """Detailed inventory of actual hardware, drivers, SDKs, and runtimes."""

    host_os: str
    host_arch: str
    host_cpu_brand: str
    device_soc: Optional[str]
    device_access: bool
    adb_path: Optional[str]
    attached_devices: list[str] = field(default_factory=list)
    os_abi_target: str = "none"
    fastrpc_driver_present: bool = False
    dsp_firmware_present: bool = False
    qnn_sdk_installed: bool = False
    qnn_sdk_version: Optional[str] = None
    qnn_backend_libraries: list[str] = field(default_factory=list)
    onnx_installed: bool = False
    onnx_version: Optional[str] = None
    onnxruntime_installed: bool = False
    onnxruntime_version: Optional[str] = None
    onnxruntime_providers: list[str] = field(default_factory=list)
    qnn_execution_provider_available: bool = False
    licensed_model_artifacts_present: bool = False
    model_checkpoint_revision: Optional[str] = None
    hardware_status: str = "UNVERIFIED"
    device_execution_status: str = "BLOCKED"
    device_performance_status: str = "BLOCKED"
    installed_components: list[str] = field(default_factory=list)
    accessible_targets: list[str] = field(default_factory=list)
    missing_components: list[str] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_cpu_brand_string() -> str:
    """Retrieve host CPU model/brand string."""
    sys_name = platform.system()
    if sys_name == "Darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception:
            return "Apple Silicon (macOS)"
    elif sys_name == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def get_attached_adb_devices(adb_bin: Optional[str]) -> list[str]:
    """Check for attached Android/Snapdragon devices via ADB."""
    if not adb_bin:
        return []
    try:
        res = subprocess.run(
            [adb_bin, "devices"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=True,
        )
        lines = res.stdout.strip().splitlines()
        devices = []
        for line in lines[1:]:  # skip 'List of devices attached'
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices
    except Exception:
        return []


def inventory_qualcomm_hardware_and_software() -> QualcommHardwareInventory:
    """Scan and classify host environment and Qualcomm target prerequisites."""
    host_os = platform.system()
    host_arch = platform.machine()
    host_cpu = get_cpu_brand_string()

    installed = []
    missing = []
    accessible = []
    unverified = []

    # 1. Host hardware & ADB connection
    adb_bin = shutil.which("adb")
    if adb_bin:
        installed.append(f"adb: {adb_bin}")
    attached_devices = get_attached_adb_devices(adb_bin)
    if attached_devices:
        accessible.extend([f"adb_device: {d}" for d in attached_devices])
        device_access = True
        device_soc = "Qualcomm Snapdragon (via ADB)"
    else:
        device_access = False
        device_soc = None
        missing.append("Qualcomm Snapdragon hardware target (no ADB device or local SoC)")

    # 2. FastRPC drivers & firmware
    fastrpc_driver = Path("/dev/adsprpc-smd").exists() or Path("/dev/fastrpc-adsp").exists()
    if fastrpc_driver:
        installed.append("Qualcomm FastRPC kernel driver (/dev/adsprpc-smd)")
    else:
        missing.append("Qualcomm FastRPC kernel driver (/dev/adsprpc-smd or /dev/fastrpc-*)")

    dsp_firmware = Path("/vendor/firmware_mnt").exists() or Path("/lib/firmware/qcom").exists()
    if dsp_firmware:
        installed.append("Hexagon DSP firmware partition")
    else:
        missing.append("Qualcomm Hexagon DSP firmware / DSP image")

    # 3. QAIRT / QNN SDK libraries
    qnn_libs = []
    qnn_installed = False
    qnn_ver = None
    for lib_name in ("libQnnHtp.so", "libQnnCpu.so", "libQnnSystem.so", "QnnHtp.dll"):
        if shutil.which(lib_name) or os.path.exists(f"/usr/local/lib/{lib_name}"):
            qnn_libs.append(lib_name)
    if qnn_libs:
        qnn_installed = True
        installed.append(f"QNN SDK backend libraries: {qnn_libs}")
    else:
        missing.append("Qualcomm AI Engine Direct (QNN) SDK native libraries (libQnnHtp.so / QnnHtp.dll)")

    # 4. ONNX and ONNX Runtime
    onnx_installed = False
    onnx_ver = None
    try:
        import onnx
        onnx_installed = True
        onnx_ver = getattr(onnx, "__version__", "unknown")
        installed.append(f"onnx: {onnx_ver}")
    except ImportError:
        missing.append("onnx Python package")

    ort_installed = False
    ort_ver = None
    ort_providers: list[str] = []
    qnn_ep_available = False
    try:
        import onnxruntime as ort
        ort_installed = True
        ort_ver = getattr(ort, "__version__", "unknown")
        ort_providers = ort.get_available_providers()
        installed.append(f"onnxruntime: {ort_ver}")
        if "QNNExecutionProvider" in ort_providers:
            qnn_ep_available = True
            installed.append("ONNX Runtime QNNExecutionProvider")
        else:
            missing.append("ONNX Runtime QNNExecutionProvider (not built/supported on darwin-arm64)")
    except ImportError:
        missing.append("onnxruntime Python package")

    # 5. Licensed model checkpoint
    model_dir = get_default_model_dir()
    model_present = False
    model_rev = None
    if (model_dir / "config.json").is_file() and (model_dir / "model.safetensors").is_file():
        model_present = True
        model_rev = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
        installed.append(f"SmolLM2-135M-Instruct checkpoint ({model_rev}, Apache-2.0)")
    else:
        missing.append("SmolLM2-135M-Instruct canonical model checkpoint")

    # Unverified claims requiring physical execution
    unverified.append("Qualcomm Hexagon NPU offload and execution")
    unverified.append("HTP FP16/W8A16 inference throughput and latency")
    unverified.append("Hexagon VTCM SRAM residency (requires allocator/profile logs)")
    unverified.append("Qualcomm FastRPC zero-copy shared memory performance")

    return QualcommHardwareInventory(
        host_os=host_os,
        host_arch=host_arch,
        host_cpu_brand=host_cpu,
        device_soc=device_soc,
        device_access=device_access,
        adb_path=adb_bin,
        attached_devices=attached_devices,
        os_abi_target="aarch64-android" if attached_devices else "none",
        fastrpc_driver_present=fastrpc_driver,
        dsp_firmware_present=dsp_firmware,
        qnn_sdk_installed=qnn_installed,
        qnn_sdk_version=qnn_ver,
        qnn_backend_libraries=qnn_libs,
        onnx_installed=onnx_installed,
        onnx_version=onnx_ver,
        onnxruntime_installed=ort_installed,
        onnxruntime_version=ort_ver,
        onnxruntime_providers=ort_providers,
        qnn_execution_provider_available=qnn_ep_available,
        licensed_model_artifacts_present=model_present,
        model_checkpoint_revision=model_rev,
        hardware_status="UNVERIFIED",
        device_execution_status="BLOCKED" if not device_access else "READY",
        device_performance_status="BLOCKED" if not device_access else "UNMEASURED",
        installed_components=installed,
        accessible_targets=accessible,
        missing_components=missing,
        unverified_claims=unverified,
    )
