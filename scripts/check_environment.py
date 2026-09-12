"""Verify the local Python and PyTorch CUDA environment."""

from __future__ import annotations

import json
import platform
import sys

import torch


def main() -> None:
    """Print environment facts and perform a real GPU matrix multiplication."""
    cuda_available = torch.cuda.is_available()
    report: dict[str, object] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
    }

    if not cuda_available:
        print(json.dumps(report, indent=2))
        raise SystemExit("CUDA is not available to PyTorch.")

    device = torch.device("cuda:0")
    report.update(
        {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_compute_capability": ".".join(
                map(str, torch.cuda.get_device_capability(device))
            ),
        }
    )

    generator = torch.Generator(device=device).manual_seed(2026)
    matrix = torch.randn((1024, 1024), generator=generator, device=device)
    product = matrix @ matrix.T
    torch.cuda.synchronize(device)
    checksum = float(product.diagonal().mean().cpu())
    report["gpu_matmul_checksum"] = round(checksum, 6)
    report["gpu_matmul_finite"] = bool(torch.isfinite(product).all().item())

    print(json.dumps(report, indent=2))
    if not report["gpu_matmul_finite"]:
        raise SystemExit("GPU matrix multiplication produced non-finite values.")


if __name__ == "__main__":
    main()
