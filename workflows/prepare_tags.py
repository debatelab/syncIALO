"Script for preparing and splitting tags data"

import os
import random

from loguru import logger
from prefect import flow, task


ALL_TAGS_PATH = "./data/tags.txt"
UNIVERSAL_TAGS_PATH = "./data/universal_tags.txt"
EVAL_TAGS_PATH = "./data/eval_tags.txt"
TEST_TAGS_PATH = "./data/test_tags.txt"


@task
def check_tag_splits_exist():
    """
    Check if tags splits exist
    """
    return os.path.exists(UNIVERSAL_TAGS_PATH) and os.path.exists(EVAL_TAGS_PATH) and os.path.exists(TEST_TAGS_PATH)


@task
def split_tags():
    if os.path.exists(ALL_TAGS_PATH):
        with open(ALL_TAGS_PATH) as file:
            tags = [line.rstrip() for line in file]
        logger.info(f"Loaded {len(tags)} tags.")

        random_tagging = random.Random(42)
        tags_universal = tags.copy()
        tags_eval = random_tagging.sample(tags_universal, k=10)
        tags_universal = [t for t in tags_universal if t not in tags_eval]
        tags_test = random_tagging.sample(tags_universal, k=10)
        tags_universal = [t for t in tags_universal if t not in tags_test]

        assert set(tags) == set(tags_universal) | set(tags_test) | set(tags_eval)

        with open(UNIVERSAL_TAGS_PATH, "w") as file:
            for tag in tags_universal:
                file.write(f"{tag}\n")
            logger.info("Created universal tags split.")
        with open(EVAL_TAGS_PATH, "w") as file:
            for tag in tags_eval:
                file.write(f"{tag}\n")
            logger.info("Created eval tags split.")
        with open(TEST_TAGS_PATH, "w") as file:
            for tag in tags_test:
                file.write(f"{tag}\n")
            logger.info("Created test tags split.")
    else:
        raise Exception("No tags file found.")


@task
def load_tags():
    with open(UNIVERSAL_TAGS_PATH) as file:
        tags_universal = [line.rstrip() for line in file]
        logger.info(f"Found {len(tags_universal)} universal tags.")
    with open(EVAL_TAGS_PATH) as file:
        tags_eval = [line.rstrip() for line in file]
        logger.info(f"Found {len(tags_eval)} eval tags.")
    with open(TEST_TAGS_PATH) as file:
        tags_test = [line.rstrip() for line in file]
        logger.info(f"Found {len(tags_test)} test tags.")


@flow(log_prints=True)
def prepare_tags():
    """
    Prepare and split tags data
    """

    if not check_tag_splits_exist():
        split_tags()

    load_tags()


if __name__ == "__main__":
    # prepare_tags.serve(name="prepare_tags-deployment")
    prepare_tags()
