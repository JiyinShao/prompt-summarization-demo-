import json

def load_dataset(name="cnn"):
    path_map = {
        "cnn": "data/input_texts.json",
        "xsum": "data/xsum_sample.json"
    }

    path = path_map.get(name.lower())
    if not path:
        raise ValueError(f"Unsupported dataset: {name}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
