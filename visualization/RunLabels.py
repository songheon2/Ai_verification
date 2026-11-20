"""Consistent model/property labels without changing model identifiers."""
import re


def _stem(path):
    if not isinstance(path, (str, bytes)) and path is not None:
        path = str(path)
    if not path or isinstance(path, bytes):
        return ""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".gz"):
        name = name[:-3]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name.strip()


def property_display_name(vnnlib_path, model_path=None):
    name = _stem(vnnlib_path)
    if not name or name.lower() in {"property", "spec", "specification"}:
        return None
    if name.casefold() == _stem(model_path).casefold():
        return None  # A model-named VNNLIB adds no distinguishing property label.
    numbered = re.fullmatch(r"prop[_\-\s]*(\d+)", name, re.IGNORECASE)
    return "prop" + numbered[1].lstrip("0").zfill(1) if numbered else name


def network_display_name(context):
    name = context.get("model_name")
    if not name:
        return None
    prop = context.get("property_name")
    return f"{name} ({prop})" if isinstance(prop, str) and prop.strip() else str(name)
