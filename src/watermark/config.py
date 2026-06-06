"""Application configuration via pydantic-settings.

Environment variables override defaults: prefix `WATERMARK_`.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-level settings.

    All paths are absolute. The default `project_root` is two parents above this
    file: `src/watermark/config.py` -> `src/watermark/` -> `src/` -> project root.
    """

    model_config = SettingsConfigDict(env_prefix="WATERMARK_", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    frames_dir: Path = Path(__file__).resolve().parents[2] / "frames"
    frames_in_dir: Path = Path(__file__).resolve().parents[2] / "frames_in"
    frames_out_dir: Path = Path(__file__).resolve().parents[2] / "frames_out"
    mask_path: Path = Path(__file__).resolve().parents[2] / "mask.png"
    alpha_map_path: Path = Path(__file__).resolve().parents[2] / "data" / "gemini_alpha_dual_96_from_grey.npy"
    logo_map_path: Path = Path(__file__).resolve().parents[2] / "data" / "gemini_logo_rgb_96_from_grey.npy"

    source_video: Path = Path(__file__).resolve().parents[2] / "omni.mp4"
    output_video: Path = Path(__file__).resolve().parents[2] / "omni_clean.mp4"

    device: str = "mps"  # one of: mps, cpu, cuda


settings = Settings()
