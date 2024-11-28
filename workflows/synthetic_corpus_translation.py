"Script for generating synthetic corpus"

import aiofiles
import argparse
import asyncio
import json
import dotenv
import enum
from pathlib import Path
import yaml
import ujson

from huggingface_hub import HfApi
from loguru import logger
import networkx as nx
from pydantic import BaseModel
from syncialo.translation import Language, translate_argmap


_BATCH_SIZE = 10

_TMP_DEBATE_FILE = "to-be-translated.json"

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        default="DebateLabKIT/syncialo-raw",
        type=str,
        help="Source dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="path to output dir"
    )
    parser.add_argument(
        "--upload_hub", action="store_true", default=True, help="Upload dataset to hub"
    )
    parser.add_argument("--hf-token", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct",
        help="Model to use for translation",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Base url of inference endpoint",
    )
    parser.add_argument(
        "--corpus-uid",
        type=str,
        required=True,
        help="Unique identifier for the corpus to translate",
    )
    parser.add_argument(
        "--source-language",
        type=str,
        default="EN",
        help="Language of source corpus",
        choices=[lang.name for lang in Language],
    )
    parser.add_argument(
        "--target-language",
        type=str,
        required=True,
        help="Language of target corpus",
        choices=[lang.name for lang in Language],
    )
    parser.add_argument(
        "--failed-to-complete-flag", type=str, help="Remove flag after completion"
    )
    args = parser.parse_args()
    return args


def create_corpus_dir(**kwargs) -> tuple[Path, Path]:
    """
    download source datasets (snapshot)
    and create new subdirectory for the translated dataset
    """
    source_path: Path = Path(kwargs["output_dir"]) / "data" / kwargs["corpus_uid"]
    if source_path.exists():
        logger.info(f"Found existing corpus directory {source_path}.")
    else:
        hfapi = HfApi(token=kwargs["hf_token"])
        hfapi.snapshot_download(
            repo_id=kwargs["source_dataset"],
            repo_type="dataset",
            allow_patterns=[f"data/{kwargs['corpus_uid']}/**/*",f"data/{kwargs['corpus_uid']}/*.yaml"],
            local_dir=kwargs["output_dir"],
        )

    target_corpus_uid = kwargs['target_corpus_uid']
    target_path: Path = Path(kwargs["output_dir"]) / "data" / target_corpus_uid
    if target_path.exists():
        logger.info(
            f"Found existing corpus directory {target_path} for translation. Will resume."
        )
    else:
        logger.info(f"Creating new corpus directory {target_path} for translation.")
        target_path.mkdir(parents=True)
        configs = yaml.safe_load((source_path / "config.yaml").read_text())
        configs["corpus_uid"] = target_corpus_uid
        configs["translated_from"] = kwargs["corpus_uid"]
        configs["translation_model"] = kwargs["model"]
        (target_path / "config.yaml").write_text(yaml.dump(configs))

    return source_path, target_path


def add_all_debate_configs(**kwargs):
    """
    creates the target debate configurations
    """

    for split in SPLIT:
        for source_config_path in Path(kwargs["source_path"] / split.value).glob("**/config.yaml"):
            debate_config = DebateConfig(
                **yaml.safe_load(source_config_path.read_text())
            )
            source_debate_uid = debate_config.debate_uid
            debate_config.corpus_uid = kwargs["target_corpus_uid"]
            debate_config.debate_uid = f"{source_debate_uid}-{kwargs['target_language']}"
            relative_path = source_config_path.relative_to(kwargs["source_path"])            
            target_config_path = Path(kwargs["target_path"] / (str(relative_path)).replace(source_debate_uid, debate_config.debate_uid))
            if target_config_path.exists():
                continue
            target_config_path.parent.mkdir(parents=True, exist_ok=True)
            target_config_path.write_text(yaml.dump(debate_config.model_dump()))
            source_json_path = next(source_config_path.parent.glob("*.json"))
            target_json_path = target_config_path.parent / _TMP_DEBATE_FILE
            target_json_path.write_text(source_json_path.read_text())


def get_missing_debates(**kwargs):
    """
    yields debate paths in the corpus for which debates haven't been translated yet
    as indicated by the presence of a _TMP_DEBATE_FILE file
    """

    for split in [SPLIT.TRAIN, SPLIT.EVAL, SPLIT.TEST]:
        if not (kwargs["target_path"] / split.value).exists():
            continue
        for debate_path in (kwargs["target_path"] / split.value).iterdir():
            if not debate_path.is_dir():
                continue
            config_path: Path = debate_path / "config.yaml"
            if config_path.exists() and any(_TMP_DEBATE_FILE in str(p) for p in debate_path.glob("*.json")):
                yield debate_path


async def translate_single_debate(debate_path: Path, **kwargs) -> nx.DiGraph | None:
    """
    translates a debate
    """
    debate_file_path = debate_path / _TMP_DEBATE_FILE
    if not debate_file_path.exists():
        msg = f"Debate file missing: {str(debate_file_path)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    async with aiofiles.open(debate_file_path, mode='r') as f:
        content = await f.read()
    
    source_argmap = nx.node_link_graph(json.loads(content))
    translated_argmap = await translate_argmap(source_argmap, **kwargs)
    return translated_argmap

def save_debates_in_corpus(
    debate_paths: list[Path], debates: list[nx.DiGraph | Exception], **kwargs
):
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
        if isinstance(debate, Exception):
            logger.error(f"Failed to translate debate {debate_path}: {str(debate)}")
            continue
        debate_config = DebateConfig(
            **yaml.safe_load((debate_path / "config.yaml").read_text())
        )
        node_link_data = nx.node_link_data(debate)
        with open(
            debate_path / f"node_link_data-{debate_config.debate_uid}.json", "w"
        ) as f:
            ujson.dump(node_link_data, f)
        (debate_path / _TMP_DEBATE_FILE).unlink()


async def translate_all_debates(**kwargs):
    """
    translates all debates in the corpus
    """

    while True:
        missing_debates = get_missing_debates(**kwargs)
        debate_paths: list[Path] = [
            next(missing_debates, None) for _ in range(_BATCH_SIZE)
        ]
        debate_paths = [p for p in debate_paths if p]
        logger.debug(f"Next {len(debate_paths)} missing debates: {debate_paths}")
        if not debate_paths:
            break
        coros = [
            translate_single_debate(debate_path=debate_path, **kwargs)
            for debate_path in debate_paths
        ]
        translated_debates = await asyncio.gather(*coros, return_exceptions=True)
        save_debates_in_corpus(
            debate_paths=debate_paths,
            debates=translated_debates,
            **kwargs
        )


def perform_sanity_checks(**kwargs) -> bool:
    """
    performs sanity checks on the generated corpus
    """

    passed = True

    if not (kwargs["target_path"] / "config.yaml").exists():
        logger.error("Config file missing for corpus.")
        passed = False

    for split in [SPLIT.TRAIN, SPLIT.EVAL, SPLIT.TEST]:
        if not (kwargs["target_path"] / split.value).exists():
            logger.error(f"No split directory {split.value}.")
            passed = False
            continue
        if len(list((kwargs["target_path"] / split.value).iterdir())) != len(list((kwargs["source_path"] / split.value).iterdir())):
            logger.error(
                f"Number of debates in {kwargs['target_path'] / split.value} split "
                f"does not match source {kwargs['source_path'] / split.value}: "
                f"{len(list((kwargs['target_path'] / split.value).iterdir()))} vs. "
                f"{len(list((kwargs['source_path'] / split.value).iterdir()))}"
            )
            passed = False

    for json_file in Path(kwargs["target_path"]).rglob("*.json"):
        if json_file.name == _TMP_DEBATE_FILE:
            logger.error(f"Found temporary debate file {str(json_file)}.")
            passed = False
        else:
            try:
                node_link_data = ujson.decode(json_file.read_text())
                nx.node_link_graph(node_link_data)
            except Exception as e:
                logger.error(f"Invalid debate json for {str(json_file)}: {str(e)}")
                passed = False

    if passed:
        logger.info("✅ All checks passed.")

    return passed




def upload_to_hf_hub(**kwargs):
    """
    uploads the debate corpus to Hugging Face Hub
    """
    logger.critical("HF Hub upload not implemented. Skipping.")


async def main():
    """
    Workflow for translating a synthetic corpus
    """
    args = parse_args()
    target_corpus_uid = f"{args.corpus_uid}-{args.target_language}"
    source_path, target_path = create_corpus_dir(
        **vars(args), target_corpus_uid=target_corpus_uid
    )
    add_all_debate_configs(
        source_path=source_path,
        target_path=target_path,
        target_corpus_uid=target_corpus_uid,
        **vars(args),
    )
    await translate_all_debates(
        source_path=source_path,
        target_path=target_path,
        target_corpus_uid=target_corpus_uid,
        **vars(args),
    )
    perform_sanity_checks(
        source_path=source_path,
        target_path=target_path,
        target_corpus_uid=target_corpus_uid,
        **vars(args),
    )
    if args.upload_hub:
        upload_to_hf_hub(
            source_path=source_path,
            target_path=target_path,
            target_corpus_uid=target_corpus_uid,
            **vars(args),
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
