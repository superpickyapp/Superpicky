# -*- coding: utf-8 -*-
"""
Runtime requirements manager for lightweight builds.

This module provides a unified interface for managing platform-specific runtime
dependencies across CPU, CUDA, and macOS builds. It consolidates the previously
separate requirements_runtime_*.txt files into a single Python module with
type-safe configuration access.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal


PlatformType = Literal["cpu", "cuda", "mac"]


@dataclass(frozen=True)
class RuntimeRequirements:
    """Runtime dependency configuration for a specific platform."""

    torch_version: str
    torchvision_version: str
    timm_version: str
    extra_index_urls: list[str]
    index_url: str | None = None

    @staticmethod
    def _format_pinned_requirement(package_name: str, version: str) -> str:
        """Return a pinned requirement only when a version is provided."""

        normalized_version = version.strip()
        if not normalized_version:
            return package_name
        return f"{package_name}=={normalized_version}"

    def to_requirements_list(self) -> list[str]:
        """Convert configuration to pip requirements list format."""
        requirements = []
        if self.index_url:
            requirements.append(f"--index-url {self.index_url}")
        for url in self.extra_index_urls:
            requirements.append(f"--extra-index-url {url}")
        requirements.append(self._format_pinned_requirement("torch", self.torch_version))
        requirements.append(
            self._format_pinned_requirement("torchvision", self.torchvision_version)
        )
        requirements.append(f"timm{self.timm_version}")
        return requirements

    def to_requirements_string(self) -> str:
        """Convert configuration to pip requirements file format."""
        lines = []
        if self.index_url:
            lines.append(f"--index-url {self.index_url}")
        for url in self.extra_index_urls:
            lines.append(f"--extra-index-url {url}")
        lines.append(self._format_pinned_requirement("torch", self.torch_version))
        lines.append(self._format_pinned_requirement("torchvision", self.torchvision_version))
        lines.append(f"timm{self.timm_version}")
        return "\n".join(lines)


def get_cpu_requirements() -> RuntimeRequirements:
    """Get runtime requirements for CPU builds."""
    return RuntimeRequirements(
        torch_version="2.7.1+cpu",
        torchvision_version="0.22.1+cpu",
        timm_version=">=0.9.0",
        extra_index_urls=[
            "https://mirror.nju.edu.cn/pytorch/whl/cpu/",
            "https://download.pytorch.org/whl/cpu",
        ],
    )


def get_cuda_requirements() -> RuntimeRequirements:
    """Get runtime requirements for CUDA builds."""
    return RuntimeRequirements(
        torch_version="2.7.1+cu118",
        torchvision_version="0.22.1+cu118",
        timm_version=">=0.9.0",
        extra_index_urls=[
            "https://mirror.nju.edu.cn/pytorch/whl/cu118/",
            "https://download.pytorch.org/whl/cu118",
        ],
    )


def get_mac_requirements() -> RuntimeRequirements:
    """Get runtime requirements for macOS builds."""
    return RuntimeRequirements(
        torch_version="2.8.0",
        torchvision_version="0.23.0",
        timm_version=">=0.9.0",
        extra_index_urls=[],
    )


def detect_platform() -> PlatformType:
    """Detect the current platform type."""
    if sys.platform == "darwin":
        return "mac"
    if sys.platform == "win32":
        return "cuda"
    return "cpu"


def get_runtime_requirements(platform: PlatformType | None = None) -> RuntimeRequirements:
    """
    Get runtime requirements for the specified or detected platform.

    Args:
        platform: Platform type ('cpu', 'cuda', 'mac'). If None, auto-detects.

    Returns:
        RuntimeRequirements: Platform-specific dependency configuration.

    Raises:
        ValueError: If platform type is invalid.
    """
    if platform is None:
        platform = detect_platform()

    requirements_getters = {
        "cpu": get_cpu_requirements,
        "cuda": get_cuda_requirements,
        "mac": get_mac_requirements,
    }

    getter = requirements_getters.get(platform)
    if getter is None:
        raise ValueError(f"Unsupported platform: {platform}")

    return getter()


def get_requirements_file_path(platform: PlatformType | None = None) -> str:
    """
    Get the legacy requirements file path for backward compatibility.

    This function is provided for migration purposes and should be replaced
    with direct usage of get_runtime_requirements().

    Args:
        platform: Platform type ('cpu', 'cuda', 'mac'). If None, auto-detects.

    Returns:
        str: Legacy requirements file name.
    """
    if platform is None:
        platform = detect_platform()

    file_mapping = {
        "cpu": "requirements_runtime_cpu.txt",
        "cuda": "requirements_runtime_cuda.txt",
        "mac": "requirements_runtime_mac.txt",
    }

    return file_mapping.get(platform, "requirements_runtime_cpu.txt")
