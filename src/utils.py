import yaml
from pathlib import Path

def read_yaml(path: Path) -> dict:
    """Load yaml file into dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: Path):
    """Create directory if it doesn’t exist."""
    path.mkdir(parents=True, exist_ok=True)

def save_json(data: dict, path: Path):
    """Save dictionary as JSON."""
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path: Path) -> dict:
    """Load JSON as dictionary."""
    import json
    with open(path, "r") as f:
        return json.load(f)
