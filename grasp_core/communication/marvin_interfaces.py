"""Load Apex ``marvin_msgs`` interfaces from the active robot workspace."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from pathlib import Path

from grasp_core.communication.request_ik_feedback import _running_colcon_install


def _candidate_install_roots() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("MARVIN_ROBOT_INSTALL")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for value in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not value:
            continue
        prefix = Path(value)
        # In an isolated colcon install AMENT_PREFIX_PATH contains the package
        # prefix; normalize that back to the common install root.
        candidates.append(prefix.parent if prefix.name == "marvin_msgs" else prefix)
    running = _running_colcon_install("marvin_qp_controller")
    if running is not None:
        candidates.append(running)

    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _python_interface_dir(package_prefix: Path) -> Path | None:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for relative in (
        Path("local/lib") / version / "dist-packages",
        Path("lib") / version / "site-packages",
        Path("lib") / version / "dist-packages",
    ):
        candidate = package_prefix / relative
        if (candidate / "marvin_msgs" / "__init__.py").is_file():
            return candidate
    return None


def _preload_type_support(package_prefix: Path) -> None:
    """Load package-local libraries when LD_LIBRARY_PATH missed the overlay."""
    library_dir = package_prefix / "lib"
    for name in (
        "libmarvin_msgs__rosidl_generator_c.so",
        "libmarvin_msgs__rosidl_generator_py.so",
        "libmarvin_msgs__rosidl_typesupport_fastrtps_c.so",
        "libmarvin_msgs__rosidl_typesupport_introspection_c.so",
        "libmarvin_msgs__rosidl_typesupport_c.so",
    ):
        library = library_dir / name
        if library.is_file():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _validate_type_support(JointcmdArm, Jointfeedback, Int) -> None:
    JointcmdArm.__class__.__import_type_support__()
    Jointfeedback.__class__.__import_type_support__()
    Int.__class__.__import_type_support__()


def _clear_partial_import() -> None:
    for name in tuple(sys.modules):
        if name == "marvin_msgs" or name.startswith("marvin_msgs."):
            sys.modules.pop(name, None)


def load_marvin_interfaces():
    """Return the three custom ROS interface types used by the wave action."""
    try:
        from marvin_msgs.msg import JointcmdArm, Jointfeedback
        from marvin_msgs.srv import Int

        _validate_type_support(JointcmdArm, Jointfeedback, Int)
        return JointcmdArm, Jointfeedback, Int
    except ModuleNotFoundError as initial_error:
        if not (initial_error.name or "").startswith("marvin_msgs"):
            raise
    except Exception:
        # The Python package can be visible while its package-local shared
        # libraries are absent from the process loader path. Retry below after
        # loading them from the matching robot install.
        pass

    checked: list[str] = []
    for install_root in _candidate_install_roots():
        package_prefix = install_root / "marvin_msgs"
        interface_dir = _python_interface_dir(package_prefix)
        checked.append(str(package_prefix))
        if interface_dir is None:
            continue
        try:
            _preload_type_support(package_prefix)
            if str(interface_dir) not in sys.path:
                sys.path.insert(0, str(interface_dir))
            importlib.invalidate_caches()
            from marvin_msgs.msg import JointcmdArm, Jointfeedback
            from marvin_msgs.srv import Int

            # Fail here with an actionable startup error instead of later when
            # rclpy tries to create the first publisher/subscription.
            _validate_type_support(JointcmdArm, Jointfeedback, Int)
            return JointcmdArm, Jointfeedback, Int
        except Exception:  # Try another matching workspace, if one exists.
            _clear_partial_import()
            continue

    detail = ", ".join(checked) or "no active robot install found"
    raise RuntimeError(
        "marvin_msgs is unavailable. Source the robot workspace before "
        "starting the app, or set MARVIN_ROBOT_INSTALL to its install "
        f"directory. Checked: {detail}"
    )


__all__ = ["load_marvin_interfaces"]
