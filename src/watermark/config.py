"""Application configuration via pydantic-settings.

Environment variables override JSON defaults: prefix `WATERMARK_`.

Source priority (later wins): init args < config.json < env vars.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    """Project-level settings.

    All paths are absolute. The default `project_root` is two parents above this
    file: `src/watermark/config.py` -> `src/watermark/` -> `src/` -> project root.

    Tweakable runtime values live in `config.json` at the project root.
    """

    model_config = SettingsConfigDict(
        json_file="config.json",
        json_file_encoding="utf-8",
        env_prefix="WATERMARK_",
        extra="ignore",
    )

    # ---- paths ----
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

    # ---- alpha verification tolerances ----
    # PNG-layer round-trip: max per-pixel uint8 diff in alpha region.
    alpha_png_max_diff_tolerance: float = 80.0
    # PNG-layer round-trip: mean per-pixel diff in alpha region.
    alpha_png_mean_diff_tolerance: float = 12.0
    # H.264-layer round-trip: max per-pixel diff after re-encode.
    alpha_h264_pixel_tolerance: float = 85.0

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, JsonConfigSettingsSource(settings_cls), env_settings, file_secret_settings)


settings = Settings()
