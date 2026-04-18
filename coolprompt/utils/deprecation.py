import warnings


def warn_deprecated(name: str) -> None:
    """
    Show a deprecation warning.

    This helper is used to mark old methods as deprecated and not recommended for use.

    Args:
        name: Name of the deprecated method.
    """
    warnings.warn(
        f"{name} method is deprecated and not recommended for use.",
        FutureWarning,
        stacklevel=1,
    )