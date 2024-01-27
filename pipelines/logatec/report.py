import click
from pathlib import Path


@click.command()
@click.argument("reports_path_pattern", nargs=-1, type=click.Path(path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path))
def cli():
    pass

    # TODO: Merge all reports into one.


if __name__ == "__main__":
    cli()
