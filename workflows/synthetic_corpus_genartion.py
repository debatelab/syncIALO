"Script for generating synthetic corpus"

import asyncio
import os
import random

from loguru import logger
from prefect import flow, task


_BATCH_SIZE = 10


@task
def create_corpus_dir(**kwargs):
    """
    creates the corpus directory, if it does not exist
    adds split subdirectories
    """
    path = "./data/corpus"
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


@flow(log_prints=True)
async def synthetic_corpus_generation(**kwargs):
    """
    Workflow for generating a synthetic corpus
    """
    path = create_corpus_dir(**kwargs)
    add_debate_configs(path=path, **kwargs)
    add_tags(path=path, **kwargs)
    add_motions(path=path, **kwargs)
    await add_debates(path=path, **kwargs)


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
        )
    )
