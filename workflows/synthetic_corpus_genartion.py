"Script for generating synthetic corpus"

import os
import random

from loguru import logger
from prefect import flow, task


@task
def create_corpus_dir():
    """
    creates the corpus directory, if it does not exist
    adds split subdirectories
    """
    path = "./data/corpus"
    return path


@task
def add_debate_configs():
    """
    creates the debate configurations
    """


@task
def add_tags():
    """
    adds tags to the corpus' debates
    """


@task
def add_motions():
    """
    adds topics and motions to the corpus' debates
    """


@task
def add_debates():
    """
    adds debates to the corpus
    """


@flow(log_prints=True)
def synthetic_corpus_generation(**kwargs):
    """
    Workflow for generating a synthetic corpus
    """
    path = create_corpus_dir()
    add_debate_configs(path=path, **kwargs)
    add_tags(path=path, **kwargs)
    add_motions(path=path, **kwargs)
    add_debates(path=path, **kwargs)


if __name__ == "__main__":
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
