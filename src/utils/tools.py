from typing import Any


def enable_bracket_access(cls):
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{type(self).__name__}' 没有属性 '{key}'")

    def __setitem__(self, key, value):
        setattr(self, key, value)

    cls.__getitem__ = __getitem__
    cls.__setitem__ = __setitem__

    return cls


def extract_answer(response: str) -> str:
    import re

    pattern = re.compile(r"<answer>([^<]*)</answer>")
    matcher = pattern.findall(response)
    if len(matcher) < 1:
        raise ValueError(f'No answer found in "{response}"')
    return matcher[-1]


def update_dict(
    total_dict: dict[str, Any], item_dict: dict[str, Any]
) -> dict[str, Any]:
    def update_dict(
        total_dict: dict[str, Any], item_dict: dict[str, Any]
    ) -> dict[str, Any]:
        for key, value in total_dict.items():
            if key in item_dict:
                total_dict[key] = item_dict[key]
            if isinstance(value, dict):
                update_dict(value, item_dict)
        return total_dict

    return update_dict(total_dict, item_dict)


def is_convertible_to_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def custom_cfgs_to_dict(key_list: str, value: Any) -> dict[str, Any]:
    """This function is used to convert the custom configurations to dict."""
    if value == "True":
        value = True
    elif value == "False":
        value = False
    elif value.isdigit():
        value = int(value)
    elif is_convertible_to_float(value):
        value = float(value)
    elif value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
        value = value.split(",")
        value = list(filter(None, value))
    elif "," in value:
        value = value.split(",")
        value = list(filter(None, value))
    else:
        value = str(value)
    keys_split = key_list.replace("-", "_").split(":")
    return_dict = {keys_split[-1]: value}

    for key in reversed(keys_split[:-1]):
        return_dict = {key.replace("-", "_"): return_dict}
    return return_dict
