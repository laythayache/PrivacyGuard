"""Parameter validation utilities for PrivacyGuard."""

from __future__ import annotations


def validate_threshold(value: float, name: str = "threshold") -> None:
    """Validate that threshold is in range [0, 1].

    Args:
        value: Threshold value to validate
        name: Parameter name for error messages

    Raises:
        TypeError: If value is not numeric
        ValueError: If value is not in range [0, 1]
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be 0-1, got {value}")


def validate_kernel_size(value: int | tuple[int, int], name: str = "kernel_size") -> None:
    """Validate that kernel size is odd and positive.

    Args:
        value: Kernel size (int or tuple of ints)
        name: Parameter name for error messages

    Raises:
        ValueError: If kernel size is not positive and odd
    """
    sizes = (value,) if isinstance(value, int) else value
    for sz in sizes:
        if sz <= 0 or sz % 2 == 0:
            raise ValueError(f"{name} must be positive and odd, got {value}")


def validate_positive(value: int | float, name: str = "value") -> None:
    """Validate that value is positive.

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: int | float, name: str = "value") -> None:
    """Validate that value is non-negative.

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_color(color: tuple[int, int, int], name: str = "color") -> None:
    """Validate BGR color tuple.

    Args:
        color: BGR color tuple
        name: Parameter name for error messages

    Raises:
        ValueError: If color tuple is invalid or channel values out of range
    """
    if len(color) != 3:
        raise ValueError(f"{name} must be 3-tuple (B, G, R), got {len(color)} values")
    for channel in color:
        if not (0 <= channel <= 255):
            raise ValueError(f"{name} channels must be 0-255, got {color}")
