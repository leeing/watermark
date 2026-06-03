"""Click CLI entry point. Registers subcommands."""

import click

from watermark.commands.detect import detect_cmd
from watermark.commands.mask import mask_cmd
from watermark.commands.remove import remove_cmd


@click.group()
@click.version_option(package_name="watermark")
def cli() -> None:
    """Remove Gemini ✦ watermark from omni.mp4 using local LaMa inpainting."""


cli.add_command(detect_cmd, name="detect")
cli.add_command(mask_cmd, name="mask")
cli.add_command(remove_cmd, name="remove")


if __name__ == "__main__":
    cli()
