"Script for generating synthetic corpus"

import asyncio
import os
from pathlib import Path
import random
import yaml

from loguru import logger
from prefect import flow, task


_BATCH_SIZE = 10


def check_kwargs(**kwargs):
    """
    checks if the provided kwargs are valid
    """
    if "corpus_uid" not in kwargs:
        raise ValueError("corpus_uid is required")
    if "tags_per_cluster" not in kwargs:
        raise ValueError("tags_per_cluster is required")
    if "debates_per_tag_cluster" not in kwargs:
        raise ValueError("debates_per_tag_cluster is required")
    if "train_split_size" not in kwargs:
        raise ValueError("train_split_size is required")
    if "eval_split_size" not in kwargs:
        raise ValueError("eval_split_size is required")
    if "test_split_size" not in kwargs:
        raise ValueError("test_split_size is required")
    if "degree_configs" not in kwargs:
        raise ValueError("degree_configs is required")
    if "output_dir" not in kwargs:
        raise ValueError("output_dir is required")


@task
def create_corpus_dir(**kwargs) -> Path:
    """
    creates the corpus directory, if it does not exist
    adds split subdirectories
    """
    path: Path = Path(kwargs["output_dir"]) / kwargs["corpus_uid"]
    train_path = path / "train"
    eval_path = path / "eval"
    test_path = path / "test"
    config_path = path / "config.yaml"
    if path.exists():
        if not (train_path.exists() and eval_path.exists() and test_path.exists()):
            logger.error("Corpus directory exists, but split directories are missing")
            raise ValueError("Corpus directory exists, but split directories are missing")
        if not config_path.exists():
            logger.error("Corpus directory exists, but config file is missing")
            raise ValueError("Corpus directory exists, but config file is missing")
        configs = yaml.safe_load(config_path.read_text())
        if configs != kwargs:
            logger.error("Corpus directory exists, but config file is different "
                         "from current config. Will not resume generation to avoid "
                         "inconsistency.")
            raise ValueError("Corpus directory exists, but config file is different")
        logger.info("Found existing corpus directory. Will resume generation.")
    else:
        train_path.mkdir(parents=True)
        eval_path.mkdir(parents=True)
        test_path.mkdir(parents=True)
        config_path.write_text(yaml.dump(kwargs))
    return path


@task
def add_all_debate_configs(**kwargs):
    """
    creates the debate configurations
    """


@task
def add_all_tags(**kwargs):
    """
    adds tags to the corpus' debates
    """


@task
def add_all_motions(**kwargs):
    """
    adds topics and motions to the corpus' debates
    """


def get_missing_debates(**kwargs):
    """
    returns a list of debate ids that are missing in the corpus
    """


@task
async def generate_single_debate(**kwargs):
    """
    generates a debate
    """


@task
def save_debates_in_corpus(debates, **kwargs):
    """
    adds a debate to the corpus
    """


async def add_all_debates(**kwargs):
    """
    adds debates to the corpus
    """
    while True:
        batch_ids = get_missing_debates(**kwargs)[:_BATCH_SIZE]
        if not batch_ids:
            break
        coros = [generate_single_debate(id) for id in batch_ids]
        save_debates_in_corpus(
            ids=batch_ids,
            debates=await asyncio.gather(*coros),
            **kwargs
        )


@task
def upload_to_hf_hub(**kwargs):
    """
    uploads the debate corpus to Hugging Face Hub
    """


@flow(log_prints=True)
async def synthetic_corpus_generation(**kwargs):
    """
    Workflow for generating a synthetic corpus
    """
    check_kwargs(**kwargs)
    path = create_corpus_dir(**kwargs)
    add_all_debate_configs(path=path, **kwargs)
    add_all_tags(path=path, **kwargs)
    add_all_motions(path=path, **kwargs)
    await add_all_debates(path=path, **kwargs)
    if "hf_hub" in kwargs:
        upload_to_hf_hub(path=path, **kwargs)


if __name__ == "__main__":
    asyncio.run(
        synthetic_corpus_generation(
            corpus_uid="synthetic_corpus-001",
            tags_per_cluster=8,
            debates_per_tag_cluster=10,
            train_split_size=1000,
            eval_split_size=50,
            test_split_size=50,
            degree_configs=[
                [6, 6, 1, 0],
                [5, 5, 2, 0],
                [3, 2, 2, 1, 1, 0],
                [4, 3, 2, 1, 0],
                [3, 4, 2, 1, 0],
            ],
            output_dir="./output"
        )
    )
