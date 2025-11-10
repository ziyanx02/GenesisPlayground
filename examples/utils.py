from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.ticker as ticker
import numpy as np
import yaml


def align_hf_arrays_vel_from_logger(
    t_ns: np.ndarray,  # int64 nanoseconds from logger.get()
    dq_arr: np.ndarray,  # (N, nj)
    dof_idx: int,  # which joint to plot/compare
    target_pos: Sequence[float],  # controller targets at control_dt
    control_dt: float = 0.02,
    out_rate_hz: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns HF measured joint velocity (dq_arr[:, dof_idx]) and the control-target
    velocity (d/dt of target_pos) onto a common output grid.

    Returns:
        t_out [s], target_vel_out, dq_out  (all shape (M,))
    """
    # basic guards
    if t_ns.size == 0 or dq_arr.size == 0 or len(target_pos) == 0:
        return np.array([]), np.array([]), np.array([])

    # 1) HF time in seconds relative to first sample (monotonic-ish)
    t_hf = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
    dq_joint = np.asarray(dq_arr[:, dof_idx], dtype=np.float64)

    # 2) Dedup/sort guard (rare but safe)
    order = np.argsort(t_hf)
    t_hf = t_hf[order]
    dq_joint = dq_joint[order]
    keep = np.r_[True, np.diff(t_hf) > 0.0]
    t_hf = t_hf[keep]
    dq_joint = dq_joint[keep]

    # 3) Controller time base (starts at 0 by convention)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    N = target_pos.shape[0]
    t_ctrl = np.arange(N, dtype=np.float64) * float(control_dt)

    # 4) Target velocity from target_pos
    target_vel_ctrl = np.gradient(target_pos, float(control_dt), edge_order=2)

    # 5) Output grid over the overlap
    dt_out = 1.0 / float(out_rate_hz)
    t_start = max(t_ctrl[0], t_hf[0])
    t_end = min(t_ctrl[-1], t_hf[-1])
    if not (t_end > t_start):
        return np.array([]), np.array([]), np.array([])

    t_out = np.arange(t_start, t_end, dt_out, dtype=np.float64)

    # 6) Interpolate both streams onto t_out
    target_vel_out = np.interp(t_out, t_ctrl, target_vel_ctrl)
    dq_out = np.interp(t_out, t_hf, dq_joint)

    return t_out, target_vel_out, dq_out


def plot_metric_on_axis(
    ax: Any,
    steps: Any,
    data_lists: list[list[float]],
    labels: list[str],
    ylabel: str,
    title: str,
    yscale: str = "log",
    xlabel: str | None = None,
    show_mean: bool = True,
) -> None:
    """Plot multiple data series on a single axis with optional log scale.

    Args:
        ax: Matplotlib axis to plot on
        steps: X-axis values (step numbers)
        data_lists: List of data arrays to plot (e.g., [mean_data, max_data])
        labels: List of labels for each data series
        ylabel: Label for y-axis
        title: Title for the subplot
        yscale: Scale for y-axis ('log' or 'linear')
        xlabel: Label for x-axis (optional)
        show_mean: Whether to display mean values as text (default: True)
    """
    # Plot each data series and their means
    for data, label in zip(data_lists, labels, strict=False):
        line = ax.plot(steps, data, label=label, linewidth=1.5, alpha=0.8)[0]

        # Add horizontal dotted line for mean if requested
        if show_mean:
            data_mean = np.mean(data)
            ax.axhline(
                y=data_mean,
                color=line.get_color(),
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label=f"mean: {data_mean:.4f}",
            )

    # Set scale
    ax.set_yscale(yscale)

    # Set ticks and formatter for log scale
    if yscale == "log":
        # Define all possible tick locations
        all_yticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

        # Filter ticks based on data range
        all_data = np.concatenate([np.array(d) for d in data_lists])
        all_data = all_data[all_data > 0]  # Filter out zeros
        if len(all_data) > 0:
            data_min = np.min(all_data)
            data_max = np.max(all_data)
            # Only include ticks within data range
            filtered_ticks = [tick for tick in all_yticks if data_min <= tick <= data_max]
            if not filtered_ticks:
                # If no ticks in range, expand slightly
                filtered_ticks = [
                    tick for tick in all_yticks if tick >= data_min * 0.5 and tick <= data_max * 2
                ][:5]
            ax.set_yticks(filtered_ticks)

        # Use decimal formatter
        formatter = ticker.FuncFormatter(lambda y, _: f"{y:.10g}")
        ax.yaxis.set_major_formatter(formatter)

    # Set labels and title
    ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add legend and grid
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")


def parse_scalar(value: Any) -> Any:
    """Best-effort parse of CLI string into bool/int/float, else return as-is."""
    if isinstance(value, int | float | bool):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except ValueError:
            return value
    return value


def strip_prefixes(path: str, prefixes: tuple[str, ...]) -> str:
    # repeatedly strip any matching prefix until none match
    changed = True
    while changed:
        changed = False
        for pref in prefixes:
            if path.startswith(pref):
                path = path[len(pref) :]
                changed = True
    return path


def tokenize_path(path: str, prefixes: tuple[str, ...]) -> list[str]:
    """Split a dot/bracket path into tokens after removing known prefixes."""
    path = strip_prefixes(path, prefixes)
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if buf:
                out.append(buf)
                buf = ""
            i += 1
            continue
        if c == "[":
            if buf:
                out.append(buf)
                buf = ""
            j = path.find("]", i)
            if j == -1:
                buf += c
                i += 1
            else:
                out.append(path[i + 1 : j])
                i = j + 1
            continue
        buf += c
        i += 1
    if buf:
        out.append(buf)
    return out


def parse_sequence_like(raw: Any, target_example: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    parts = [p.strip() for p in raw.split(",")]
    parsed = [parse_scalar(p) for p in parts]
    if isinstance(target_example, tuple):
        return tuple(parsed)
    return parsed


def deep_apply(obj: Any, tokens: list[str], raw_val: Any) -> Any:
    """Deeply apply a single override value addressed by tokens to a Pydantic model.

    Handles nested BaseModel, dicts (one-level key), and lists/tuples (by index).
    """
    from pydantic import BaseModel as _BM  # type: ignore

    if not tokens:
        return obj

    field = tokens[0]
    rest = tokens[1:]

    # getattr guard
    try:
        current = getattr(obj, field)
    except AttributeError:
        return obj

    # Leaf assignment
    if not rest:
        new_value = parse_scalar(raw_val)
        if isinstance(current, tuple | list) and isinstance(raw_val, str) and "," in raw_val:
            new_value = parse_sequence_like(raw_val, current)
        if isinstance(current, tuple) and isinstance(new_value, list):
            new_value = tuple(new_value)
        return obj.model_copy(update={field: new_value}) if isinstance(obj, _BM) else obj

    # Dict field (e.g., reward_args)
    if isinstance(current, dict):
        key = rest[0]
        new_dict = dict(current)
        if len(rest) == 1:
            new_dict[key] = parse_scalar(raw_val)
        else:
            # deeper dict nesting not supported
            new_dict[".".join(rest)] = parse_scalar(raw_val)
        return obj.model_copy(update={field: new_dict})

    # List/Tuple field
    if isinstance(current, list | tuple):
        try:
            idx = int(rest[0])
        except ValueError:
            return obj
        if idx < 0 or idx >= len(current):
            return obj
        seq_list = list(current)
        elem = seq_list[idx]
        if len(rest) == 1:
            new_elem = parse_scalar(raw_val)
        else:
            if hasattr(elem, "model_copy"):
                new_elem = deep_apply(elem, rest[1:], raw_val)
            elif isinstance(elem, dict):
                new_elem = dict(elem)
                new_elem[rest[1]] = parse_scalar(raw_val)
            else:
                new_elem = elem
        seq_list[idx] = new_elem
        new_seq = tuple(seq_list) if isinstance(current, tuple) else seq_list
        return obj.model_copy(update={field: new_seq})

    # Nested Pydantic model
    if hasattr(current, "model_copy"):
        updated_sub = deep_apply(current, rest, raw_val)
        return obj.model_copy(update={field: updated_sub})

    return obj


def apply_overrides_generic(
    base_obj: Any, overrides: dict[str, Any] | None, prefixes: tuple[str, ...]
) -> Any:
    if not overrides:
        return base_obj
    updated = base_obj
    for raw_key, raw_val in overrides.items():
        tokens = tokenize_path(raw_key, prefixes)
        updated = deep_apply(updated, tokens, raw_val)
    return updated


def _convert_to_serializable(obj: Any) -> Any:
    """Convert objects to YAML-serializable format.

    Args:
        obj: Object to convert

    Returns:
        YAML-serializable representation
    """
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convert tuple to list but mark it with a special tag
        return {"__tuple__": [_convert_to_serializable(item) for item in obj]}
    else:
        # For other types, convert to string
        return str(obj)


def _restore_tuples(obj: Any) -> Any:
    """Restore tuples from YAML data.

    Args:
        obj: Object loaded from YAML

    Returns:
        Object with tuples restored
    """
    if isinstance(obj, dict):
        if "__tuple__" in obj and len(obj) == 1:
            # This is a tuple marker
            return tuple(_restore_tuples(item) for item in obj["__tuple__"])
        else:
            return {k: _restore_tuples(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_tuples(item) for item in obj]
    else:
        return obj


def config_to_yaml(config: Any, yaml_path: str | Path) -> None:
    """Convert a config/args class (typically Pydantic model) to YAML file.

    Args:
        config: Configuration object (Pydantic model or similar)
        yaml_path: Path where YAML file should be saved

    Example:
        >>> config_to_yaml(env_args, "logs/experiment/env_args.yaml")
    """
    from pydantic import BaseModel as _BM  # type: ignore

    # Convert to dictionary
    if isinstance(config, _BM):
        config_dict = config.model_dump()
    elif hasattr(config, "__dict__"):
        config_dict = config.__dict__
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f"Cannot convert config of type {type(config)} to YAML")

    # Convert to serializable format
    serializable_dict = _convert_to_serializable(config_dict)

    # Ensure parent directory exists
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to YAML file
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(serializable_dict, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"Config saved to YAML: {yaml_path}")


def yaml_to_config(yaml_path: str | Path, config_class: type | None = None) -> Any:
    """Convert YAML file back to config/args class.

    Args:
        yaml_path: Path to YAML file
        config_class: Optional class type to instantiate. If provided and it's a
                     Pydantic model, will return an instance of that class.
                     If None, returns a dictionary.

    Returns:
        Config object (Pydantic model instance) or dictionary

    Example:
        >>> from gs_env.sim.envs.config.schema import EnvArgs
        >>> env_args = yaml_to_config("logs/experiment/env_args.yaml", EnvArgs)
    """
    from pydantic import BaseModel as _BM  # type: ignore

    # Load YAML file
    with open(yaml_path, encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Restore tuples
    config_dict = _restore_tuples(config_dict)

    # If config_class provided and is Pydantic model, instantiate it
    if config_class is not None and issubclass(config_class, _BM):
        return config_class(**config_dict)
    elif config_class is not None:
        # Try to instantiate with the dictionary
        return config_class(**config_dict)
    else:
        # Return as dictionary
        return config_dict
