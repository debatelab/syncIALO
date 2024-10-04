"Script for generating synthetic corpus"

import asyncio
import enum
import os
from pathlib import Path
import random
import yaml
import ujson

from langchain_openai import ChatOpenAI
from loguru import logger
import networkx as nx
from prefect import flow, task
from pydantic import BaseModel
from syncialo.chains.debate_design import SuggestMotionChain, SuggestTopicsChain
from syncialo.debate_builder import DebateBuilder


_BATCH_SIZE = 10

_UNIVERSAL_TAGS_PATH = "data/universal_tags.txt"
_EVAL_TAGS_PATH = "data/eval_tags.txt"
_TEST_TAGS_PATH = "data/test_tags.txt"


class SPLIT(enum.Enum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"


class DebateConfig(BaseModel):
    split: str
    corpus_uid: str
    debate_uid: str
    tags: list[str]
    topic: str
    motion: dict[str, str]
    degree_config: list[int]


def check_kwargs(**kwargs):
    """
    checks if the provided kwargs are valid
    """
    if "corpus_uid" not in kwargs:
        raise ValueError("corpus_uid is required")
    if "universal_tags_path" not in kwargs:
        raise ValueError("universal_tags_path is required")
    if not Path(kwargs["universal_tags_path"]).exists():
        raise ValueError(f"universal_tags_path does not exist: {kwargs['universal_tags_path']}")
    if "eval_tags_path" not in kwargs:
        raise ValueError("eval_tags_path is required")
    if not Path(kwargs["eval_tags_path"]).exists():
        raise ValueError(f"eval_tags_path does not exist: {kwargs['eval_tags_path']}")
    if not Path(kwargs["test_tags_path"]).exists():
        raise ValueError(f"test_tags_path does not exist: {kwargs['test_tags_path']}")
    if "test_tags_path" not in kwargs:
        raise ValueError("test_tags_path is required")
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
    if "model_kwargs" not in kwargs:
        raise ValueError("model_kwargs is required")


@task
def create_corpus_dir(**kwargs) -> Path:
    """
    creates the corpus directory, if it does not exist
    adds split subdirectories
    returns corpus base path
    """
    path: Path = Path(kwargs["output_dir"]) / kwargs["corpus_uid"]
    config_path = path / "config.yaml"
    if path.exists():
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
        path.mkdir(parents=True)
        config_path.write_text(yaml.dump(kwargs))
    return path


@task
def add_all_debate_configs(**kwargs):
    """
    creates the debate configurations
    """

    split_sizes = {
        SPLIT.TRAIN: kwargs["train_split_size"],
        SPLIT.EVAL: kwargs["eval_split_size"],
        SPLIT.TEST: kwargs["test_split_size"],
    }

    for split, split_size in split_sizes.items():
        for i in range(split_size):
            degree_config = random.choice(kwargs["degree_configs"])
            debate_uid = f"debate-{split.value}-{(i+1):04d}"
            debate_config = DebateConfig(
                split=split.value,
                corpus_uid=kwargs["corpus_uid"],
                debate_uid=debate_uid,
                tags=[],
                topic="",
                motion={},
                degree_config=degree_config,
            )
            debate_path = kwargs["path"] / split.value / f"{debate_uid}.yaml"
            config_path = debate_path / "config.yaml"
            if debate_path.exists():
                if not config_path.exists():
                    msg = (
                        f"Debate directory {str(debate_path)} exists, but config file is missing. "
                        f"Please delete the directory before resuming."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                continue
            config_path.write_text(yaml.dump(debate_config.model_dump()))


@task
def add_all_topics(**kwargs):
    """
    adds tags and topics to the corpus' debates
    """
    tags_per_cluster = kwargs["tags_per_cluster"]

    universal_tags = Path(kwargs["universal_tags_path"]).read_text().split("\n")
    universal_tags = [tag.rstrip() for tag in universal_tags]
    logger.debug(f"Read {len(universal_tags)} universal tags")
    eval_tags = Path(kwargs["eval_tags_path"]).read_text().split("\n")
    eval_tags = [tag.rstrip() for tag in eval_tags]
    logger.debug(f"Read {len(eval_tags)} eval tags")
    test_tags = Path(kwargs["test_tags_path"]).read_text().split("\n")
    test_tags = [tag.rstrip() for tag in test_tags]
    logger.debug(f"Read {len(test_tags)} test tags")

    chat_model = ChatOpenAI(**kwargs["model_kwargs"])
    suggest_topics_chain = SuggestTopicsChain.build(chat_model)

    def sample_tags(_split: SPLIT) -> list[str]:
        if _split == SPLIT.TRAIN:
            tags = random.sample(universal_tags, tags_per_cluster)
        if _split == SPLIT.EVAL:
            k = tags_per_cluster // 2
            tags = random.sample(eval_tags, k) + random.sample(universal_tags, tags_per_cluster - k)
        if _split == SPLIT.TEST:
            k = tags_per_cluster // 2
            tags = random.sample(test_tags, k) + random.sample(universal_tags, tags_per_cluster - k)
        else:
            raise ValueError(f"Invalid split: {_split}")
        return random.shuffle(tags)

    for split in [SPLIT.TRAIN, SPLIT.EVAL, SPLIT.TEST]:
        logger.info(f"Adding topics to debates in {split} split...")
        topic_suggestions = []
        for debate_path in (kwargs["path"] / split.value).iterdir():
            if not debate_path.is_dir():
                continue
            config_path: Path = debate_path / "config.yaml"
            if not config_path.exists():
                logger.warning(f"Config file missing for {str(debate_path)}. Skipping this directory.")
                continue
            debate_config = DebateConfig(yaml.safe_load(config_path.read_text()))
            if debate_config.topic:
                continue
            if not topic_suggestions:
                tags = sample_tags(split)
                topic_suggestions = suggest_topics_chain.invoke({
                    "tags": tags,
                    "debates_per_tag_cluster": kwargs["debates_per_tag_cluster"]
                })
            debate_config.tags = tags
            debate_config.topic = topic_suggestions.pop()["topic"]
            config_path.write_text(yaml.dump(debate_config.model_dump()))


@task
def add_all_motions(**kwargs):
    """
    adds topics and motions to the corpus' debates
    """
    chat_model = ChatOpenAI(**kwargs["model_kwargs"])
    suggest_motion_chain = SuggestMotionChain.build(chat_model)

    for split in [SPLIT.TRAIN, SPLIT.EVAL, SPLIT.TEST]:
        logger.info(f"Adding motions to debates in {split} split...")
        for debate_path in (kwargs["path"] / split.value).iterdir():
            if not debate_path.is_dir():
                continue
            config_path: Path = debate_path / "config.yaml"
            if not config_path.exists():
                logger.warning(f"Config file missing for {str(debate_path)}. Skipping this directory.")
                continue
            debate_config = DebateConfig(yaml.safe_load(config_path.read_text()))
            if debate_config.motion:
                logger.debug(f"Motion already exists for {str(debate_path)}. Skipping.")
                continue
            motion = suggest_motion_chain.invoke({
                "tags": debate_config.tags,
                "topic": debate_config.topic,
            })
            if not (isinstance(motion, dict) and "title" in motion and "motion" in motion):
                msg = f"Invalid motion suggestion for {str(debate_path)}: {motion}"
                logger.error(msg)
                raise ValueError(msg)
            debate_config.motion = {"label": motion["title"], "claim": motion["motion"]}
            config_path.write_text(yaml.dump(debate_config.model_dump()))


def get_missing_debates(**kwargs):
    """
    yields debate paths in the corpus for which debates haven't been generated yet
    """
    for split in [SPLIT.TRAIN, SPLIT.EVAL, SPLIT.TEST]:
        for debate_path in (kwargs["path"] / split.value).iterdir():
            if not debate_path.is_dir():
                continue
            config_path: Path = debate_path / "config.yaml"
            if config_path.exists() and not any(debate_path.glob("*.json")):
                yield debate_path


@task
async def generate_single_debate(debate_path: Path, **kwargs) -> nx.DiGraph:
    """
    generates a debate
    """
    tags_universal = Path(kwargs["universal_tags_path"]).read_text().split("\n")
    debate_config = DebateConfig(yaml.safe_load((debate_path / "config.yaml").read_text()))

    chat_model = ChatOpenAI(**kwargs["model_kwargs"])
    debateBuilder = DebateBuilder(
        model=chat_model,
        tags_universal=tags_universal,
        tags_per_cluster=kwargs["tags_per_cluster"],
    )
    built_debate: nx.DiGraph = await debateBuilder.build_debate(
        root_claim=debate_config.motion,
        topic=debate_config.topic,
        tag_cluster=debate_config.tags,
        degree_config=debate_config.degree_config,
    )
    return built_debate


@task
def save_debates_in_corpus(debate_paths: list[Path], debates: list[nx.DiGraph], **kwargs):
    """
    adds and saves given debates to the corpus
    """
    if not len(debate_paths) == len(debates):
        msg = "Internal error: Length of debate_paths and debates must be equal."
        logger.debug("Debate paths: {debate_paths}")
        logger.debug("Debates: {debates}")
        logger.error(msg)
        raise ValueError(msg)

    for debate_path, debate in zip(debate_paths, debates):
        debate_config = DebateConfig(yaml.safe_load((debate_path / "config.yaml").read_text()))
        node_link_data = nx.node_link_data(debate)
        with open(debate_path / f"node_link_data-{debate_config.debate_uid}.json", "w") as f:
            ujson.dump(node_link_data, f)


async def add_all_debates(**kwargs):
    """
    adds all debates to the corpus
    """
    while True:
        debate_paths: list[Path] = [next(get_missing_debates(**kwargs), None) for _ in range(_BATCH_SIZE)]
        debate_paths = [p for p in debate_paths if p]
        logger.debug(f"Next {len(debate_paths)} missing debates: {debate_paths}")
        if not debate_paths:
            break
        coros = [generate_single_debate(debate_path=debate_path, **kwargs) for debate_path in debate_paths]
        save_debates_in_corpus(
            debate_paths=debate_paths,
            debates=await asyncio.gather(*coros),
            **kwargs
        )


@task
def perform_sanity_checks(**kwargs):
    """
    performs sanity checks on the generated corpus
    """

    # check for each split if all debates are generated
    debates_counter = {SPLIT.TRAIN: 0, SPLIT.EVAL: 0, SPLIT.TEST: 0}

    for split in [SPLIT.TRAIN, SPLIT.EVAL, SPLIT.TEST]:
        for debate_path in (kwargs["path"] / split.value).iterdir():
            if not debate_path.is_dir():
                continue

            config_path: Path = debate_path / "config.yaml"
            if not config_path.exists():
                logger.error(f"Config file missing for {str(debate_path)}.")
                raise ValueError(f"Config file missing for {str(debate_path)}")
            try:
                DebateConfig(yaml.safe_load(config_path.read_text()))
            except Exception as e:
                msg = f"Invalid config file for {str(debate_path)}: {str(e)}"
                logger.error(msg)
                raise ValueError(msg)

            json_files: list[Path] = list(debate_path.glob("*.json"))
            if not json_files:
                msg = f"Debate json missing for {str(debate_path)}"
                logger.error(msg)
                raise ValueError(msg)
            if len(json_files) > 1:
                msg = f"Multiple (debate?) json files found for {str(debate_path)}"
                logger.error(msg)
                raise ValueError(msg)
            try:
                node_link_data = ujson.load(json_files[0])
                nx.DiGraph(node_link_data)
            except Exception as e:
                msg = f"Invalid debate json for {str(debate_path)}: {str(e)}"
                logger.error(msg)
                raise ValueError(msg)

            debates_counter[split] += 1

    for (split, expected_size) in zip(
        [SPLIT.TRAIN, SPLIT.EVAL, SPLIT.TEST],
        [kwargs["train_split_size"], kwargs["eval_split_size"], kwargs["test_split_size"]],
    ):
        if not debates_counter[split] == expected_size:
            msg = (
                f"Invalid number of debates in {split.value} split. "
                f"Found {debates_counter[split]} debates, "
                f"expected {expected_size}."
            )
            logger.error(msg)
            raise ValueError(msg)


@task
def upload_to_hf_hub(**kwargs):
    """
    uploads the debate corpus to Hugging Face Hub
    """
    logger.warning("HF Hub upload not implemented. Skipping.")


@flow(log_prints=True)
async def synthetic_corpus_generation(**kwargs):
    """
    Workflow for generating a synthetic corpus
    """
    check_kwargs(**kwargs)
    path = create_corpus_dir(**kwargs)
    add_all_debate_configs(path=path, **kwargs)
    add_all_topics(path=path, **kwargs)
    add_all_motions(path=path, **kwargs)
    await add_all_debates(path=path, **kwargs)
    perform_sanity_checks(path=path, **kwargs)
    if "hf_hub" in kwargs:
        upload_to_hf_hub(path=path, **kwargs)


if __name__ == "__main__":
    asyncio.run(
        synthetic_corpus_generation(
            corpus_uid="synthetic_corpus-TEST-001",
            universal_tags_path=_UNIVERSAL_TAGS_PATH,
            eval_tags_path=_EVAL_TAGS_PATH,
            test_tags_path=_TEST_TAGS_PATH,
            tags_per_cluster=8,
            debates_per_tag_cluster=5,
            train_split_size=10,
            eval_split_size=0,
            test_split_size=0,
            degree_configs=[
                [2, 1, 0],
                [1, 2, 0],
            ],
            output_dir="./output",
            model_kwargs={
                "model": "tgi",
                "base_url": "http://kriton.philosophie.kit.edu:8080/v1/",
                "api_key": os.getenv("API_KEY", "no-key-required"),
            },
        )
    )

#    asyncio.run(
#        synthetic_corpus_generation(
#            corpus_uid="synthetic_corpus-001",
#            universal_tags_path=_UNIVERSAL_TAGS_PATH,
#            eval_tags_path=_EVAL_TAGS_PATH,
#            test_tags_path=_TEST_TAGS_PATH,
#            tags_per_cluster=8,
#            debates_per_tag_cluster=5,
#            train_split_size=1000,
#            eval_split_size=50,
#            test_split_size=50,
#            degree_configs=[
#                [6, 6, 1, 0],
#                [5, 5, 2, 0],
#                [3, 2, 2, 1, 1, 0],
#                [4, 3, 2, 1, 0],
#                [3, 4, 2, 1, 0],
#            ],
#            output_dir="./output",
#            model_kwargs={
#                "model": "tgi",
#                "base_url": "http://kriton.philosophie.kit.edu:8080/v1/",
#                "api_key": os.getenv("API_KEY", "no-key-required"),
#            },
#        )
#    )
