"""Backend implementations for MiniCPM-O-4.5 inference."""

from .orangepi_backend import OrangePiBackend

__all__ = ["OrangePiBackend", "create_backend"]


def create_backend(
    backend_type: str = "auto",
    **kwargs,
):
    """Create a backend instance.

    Args:
        backend_type: Backend type. Options:
            - "auto": Auto-detect (Orange Pi if on Ascend 310B, else error)
            - "orangepi": Orange Pi / Ascend 310B backend
            - "pytorch": PyTorch backend (requires torch, not available here)
        **kwargs: Backend-specific parameters

    Returns:
        Backend instance with unified interface

    Raises:
        RuntimeError: If backend_type is invalid or dependencies missing
    """
    if backend_type == "auto":
        # Try to detect Ascend environment
        try:
            import subprocess
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and "310B" in result.stdout:
                backend_type = "orangepi"
            else:
                raise RuntimeError(
                    "Auto-detection failed: No Ascend 310B NPU detected. "
                    "Please specify backend_type explicitly."
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError(
                "Auto-detection failed: npu-smi not found. "
                "Please specify backend_type explicitly."
            )

    if backend_type == "orangepi":
        return OrangePiBackend(**kwargs)
    elif backend_type == "pytorch":
        # Import pytorch backend only if requested (avoid dependency)
        try:
            from core.processors.pytorch_backend import PyTorchBackend
            return PyTorchBackend(**kwargs)
        except ImportError as e:
            raise RuntimeError(
                f"PyTorch backend not available: {e}. "
                "Install torch and MiniCPM-o dependencies."
            )
    else:
        raise ValueError(
            f"Unknown backend_type: {backend_type}. "
            f"Valid options: 'auto', 'orangepi', 'pytorch'"
        )
