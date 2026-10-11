"""Point @champion at a model version. Promotion and rollback are the same move.

    python -m pipeline.promote                 list the versions and their aliases
    python -m pipeline.promote --version 3     make version 3 the champion
    python -m pipeline.promote --rollback      swap @champion and @previous
"""
import argparse
import sys

from pipeline import settings

settings.load_env()

from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402


def version_by_alias(client, alias):
    try:
        return str(client.get_model_version_by_alias(settings.MODEL_NAME, alias).version)
    except MlflowException:
        return None


def promote(client, model_version):
    """@champion moves to `model_version`; the version it leaves becomes @previous, the rollback target."""
    current = version_by_alias(client, settings.CHAMPION)
    if current == str(model_version):
        return
    if current is not None:
        client.set_registered_model_alias(settings.MODEL_NAME, settings.PREVIOUS, current)
    client.set_registered_model_alias(settings.MODEL_NAME, settings.CHAMPION, str(model_version))


def show(client):
    aliases = client.get_registered_model(settings.MODEL_NAME).aliases
    print(settings.MODEL_NAME)
    for mv in sorted(client.search_model_versions(f"name='{settings.MODEL_NAME}'"), key=lambda v: int(v.version)):
        labels = " ".join("@" + a for a, v in aliases.items() if str(v) == str(mv.version))
        trained_until = client.get_run(mv.run_id).data.params.get("train_until", "?")
        print(f"  v{mv.version}  data up to {trained_until}  smoke check {mv.tags.get('smoke_check', '?')}  {labels}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--version", help="the version to make the champion")
    choice.add_argument("--rollback", action="store_true", help="make @previous the champion again")
    args = parser.parse_args()

    settings.setup_mlflow()
    client = MlflowClient()
    if args.rollback:
        target = version_by_alias(client, settings.PREVIOUS)
        if target is None:
            sys.exit("there is no @previous version to roll back to")
        promote(client, target)
    elif args.version:
        promote(client, args.version)
    show(client)


if __name__ == "__main__":
    main()
