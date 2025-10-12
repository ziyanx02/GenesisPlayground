from typing import Any
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np


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
        all_yticks = [0.001, 0.003,0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

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


def _dict_to_xml_element(data: dict[str, Any], root_name: str = "config") -> ET.Element:
    """Convert a dictionary to an XML Element recursively.
    
    Args:
        data: Dictionary to convert
        root_name: Name for the root element
        
    Returns:
        XML Element representation of the dictionary
    """
    root = ET.Element(root_name)
    
    def add_to_element(parent: ET.Element, key: str, value: Any) -> None:
        """Recursively add data to XML element."""
        if value is None:
            child = ET.SubElement(parent, key, type="None")
        elif isinstance(value, bool):
            child = ET.SubElement(parent, key, type="bool")
            child.text = str(value)
        elif isinstance(value, int):
            child = ET.SubElement(parent, key, type="int")
            child.text = str(value)
        elif isinstance(value, float):
            child = ET.SubElement(parent, key, type="float")
            child.text = str(value)
        elif isinstance(value, str):
            child = ET.SubElement(parent, key, type="str")
            child.text = value
        elif isinstance(value, list | tuple):
            child = ET.SubElement(parent, key, type="list" if isinstance(value, list) else "tuple")
            for i, item in enumerate(value):
                add_to_element(child, f"item_{i}", item)
        elif isinstance(value, dict):
            child = ET.SubElement(parent, key, type="dict")
            for sub_key, sub_value in value.items():
                add_to_element(child, sub_key, sub_value)
        else:
            # For other types (e.g., objects), try to convert to string
            child = ET.SubElement(parent, key, type="object")
            child.text = str(value)
    
    for key, value in data.items():
        add_to_element(root, key, value)
    
    return root


def _xml_element_to_dict(element: ET.Element) -> dict[str, Any]:
    """Convert an XML Element back to a dictionary recursively.
    
    Args:
        element: XML Element to convert
        
    Returns:
        Dictionary representation of the XML element
    """
    result: dict[str, Any] = {}
    
    def parse_element(elem: ET.Element) -> Any:
        """Recursively parse XML element to Python value."""
        elem_type = elem.get("type", "str")
        
        if elem_type == "None":
            return None
        elif elem_type == "bool":
            return elem.text == "True"
        elif elem_type == "int":
            return int(elem.text) if elem.text else 0
        elif elem_type == "float":
            return float(elem.text) if elem.text else 0.0
        elif elem_type == "str":
            return elem.text if elem.text else ""
        elif elem_type in ("list", "tuple"):
            items = []
            for child in elem:
                items.append(parse_element(child))
            return tuple(items) if elem_type == "tuple" else items
        elif elem_type == "dict":
            sub_dict = {}
            for child in elem:
                sub_dict[child.tag] = parse_element(child)
            return sub_dict
        elif elem_type == "object":
            return elem.text if elem.text else ""
        else:
            return elem.text if elem.text else ""
    
    for child in element:
        result[child.tag] = parse_element(child)
    
    return result


def config_to_xml(config: Any, xml_path: str | Path) -> None:
    """Convert a config/args class (typically Pydantic model) to XML file.
    
    Args:
        config: Configuration object (Pydantic model or similar)
        xml_path: Path where XML file should be saved
        
    Example:
        >>> config_to_xml(env_args, "logs/experiment/env_args.xml")
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
        raise ValueError(f"Cannot convert config of type {type(config)} to XML")
    
    # Convert dictionary to XML
    root = _dict_to_xml_element(config_dict, root_name="config")
    
    # Create XML tree and write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)  # Pretty print
    
    # Ensure parent directory exists
    xml_path = Path(xml_path)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Config saved to XML: {xml_path}")


def xml_to_config(xml_path: str | Path, config_class: type | None = None) -> Any:
    """Convert XML file back to config/args class.
    
    Args:
        xml_path: Path to XML file
        config_class: Optional class type to instantiate. If provided and it's a 
                     Pydantic model, will return an instance of that class.
                     If None, returns a dictionary.
        
    Returns:
        Config object (Pydantic model instance) or dictionary
        
    Example:
        >>> from gs_env.sim.envs.config.schema import EnvArgs
        >>> env_args = xml_to_config("logs/experiment/env_args.xml", EnvArgs)
    """
    from pydantic import BaseModel as _BM  # type: ignore
    
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Convert XML to dictionary
    config_dict = _xml_element_to_dict(root)
    
    # If config_class provided and is Pydantic model, instantiate it
    if config_class is not None and issubclass(config_class, _BM):
        return config_class(**config_dict)
    elif config_class is not None:
        # Try to instantiate with the dictionary
        return config_class(**config_dict)
    else:
        # Return as dictionary
        return config_dict
