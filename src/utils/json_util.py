import json


def load_jsonl(file_path: str) -> list[dict]:
    """
    Load a JSON Lines file and return a list of dictionaries.

    Args:
        file_path (str): The path to the JSON Lines file.

    Returns:
        list[dict]: A list of dictionaries loaded from the file.
    """
    data = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_json(file_path: str) -> dict:
    """
    Load a JSON file and return its content as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict, file_path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The path where the JSON file will be saved.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
