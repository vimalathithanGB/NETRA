"""
Basic Unit Test to verify project structure integrity.
"""
from pathlib import Path
import yaml


def test_directory_structure():
    base_dir = Path(__file__).resolve().parent.parent
    expected_folders = [
        "backend",
        "ai_engine",
        "models",
        "datasets",
        "training",
        "inference",
        "tracking",
        "ocr",
        "reid",
        "trajectory",
        "evaluation",
        "scripts",
        "configs",
        "tests",
    ]
    for folder in expected_folders:
        folder_path = base_dir / folder
        assert folder_path.exists() and folder_path.is_dir(), f"Missing folder: {folder}"


def test_default_config():
    base_dir = Path(__file__).resolve().parent.parent
    config_file = base_dir / "configs" / "default_config.yaml"
    assert config_file.exists(), "configs/default_config.yaml does not exist"

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "project" in config
    assert "hardware" in config
    assert "detection" in config
    assert "tracking" in config
    assert "reid" in config
