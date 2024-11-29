import enum
import json

from huggingface_hub import AsyncInferenceClient
from loguru import logger
import networkx as nx
from pydantic import BaseModel
import tenacity


class Language(enum.Enum):
    EN = "English"
    DE = "German"
    FR = "French"
    ES = "Spanish"
    PT = "Portuguese"
    IT = "Italian"
    RU = "Russian"
    JP = "Japanese"
    ZH = "Chinese"


class _ClaimModel(BaseModel):
    claim: str
    label: str


class _PremisesModel(BaseModel):
    premises: list[str]


_PROMPT_TRANSLATION_ROOT = (
    "Task: Translate to {lang}: {original_claim}\n"
    "Provide a translation of both claim and short label. "
    "Briefly check that your translation is really in {lang}, especially if "
    "it starts with words that work in English, too.\n"
)

_PROMPT_TRANSLATION_REASON = (
    "Task: Translate to {lang}: {original_claim}\n"
    "Provide a translation of both claim and short label. "
    "Briefly check that your translation is really in {lang}, especially if "
    "it starts with words that work in English, too.\n"
)

_PROMPT_TRANSLATION_PREMISES = (
    "Task: Translate to {lang}.\n\n"
    "I ask you to assist me in translating an argument from English to {lang}. "
    "We already have a translation of the argument's gist: "
    "'f{original_claim}' as 'f{translated_claim}'.\n"
    "Please translate the following list of sentences, "
    "all of which are premises of the initially mentioned argument, to {lang}: "
    "{original_premises}\n"
    "Make sure that your translations are really in {lang}."
)

_PROMPT_TRANSLATION_FORMAT = (
    "Task: Render {lang} translation in JSON format.\n"
    "Original item: {original_item}\n"
    "Translation: {free_translation}\n"
    "Please render the {lang} translation of the original item in JSON "
    "format, using the same schema as the original item."
)


@tenacity.retry(wait=tenacity.wait_random_exponential(multiplier=1, max=60), stop=tenacity.stop_after_attempt(5))
async def _translate_root(
    node_data: dict, target_language: Language, client: AsyncInferenceClient
) -> dict:
    try:
        claim = _ClaimModel(claim=node_data["claim"], label=node_data["label"])
        original_claim = claim.model_dump_json()
        prompt = _PROMPT_TRANSLATION_ROOT.format(
            lang=target_language.value, original_claim=original_claim
        )
    except Exception as e:
        logger.error(f"Failed to prepare node {node_data} due to {e}")
        raise e
    try:
        resp = await client.text_generation(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.4,
        )
    except Exception as e:
        logger.error(f"Failed to translate node {node_data} due to {e}")
        raise e

    prompt = _PROMPT_TRANSLATION_FORMAT.format(
        lang=target_language.value,
        original_item=original_claim,
        free_translation=resp,
    )
    try:
        resp = await client.text_generation(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.3,
            grammar={"type": "json", "value": _ClaimModel.model_json_schema()},
        )
    except Exception as e:
        logger.error(f"Failed to format node {node_data} due to {e}")
        raise e

    try:
        node_data = node_data.copy()
        node_data.update(json.loads(resp))
    except Exception as e:
        logger.debug(
            f"Node data: {node_data}\n"
            f"Format prompt: {prompt}\n"
            f"Response: {resp}\n"
            f"Response type: {type(resp)}\n"
        )
        logger.error(f"Failed to update node {node_data} with {resp} due to {e}")
        raise e

    logger.debug(f"Translation: {str(node_data)[:100]}...")
    return node_data


@tenacity.retry(wait=tenacity.wait_random_exponential(multiplier=1, max=60), stop=tenacity.stop_after_attempt(5))
async def _translate_reason(
    node_data: dict,
    target_language: Language,
    client: AsyncInferenceClient,
) -> dict:
    claim = _ClaimModel(claim=node_data["claim"], label=node_data["label"])
    original_claim = claim.model_dump_json()

    try:
        prompt = _PROMPT_TRANSLATION_REASON.format(
            lang=target_language.value,
            original_claim=original_claim,
        )
        resp = await client.text_generation(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.4,
        )
    except Exception as e:
        logger.error(f"Failed to translate claim gist and label {node_data} due to {e}")
        raise e

    try:
        prompt = _PROMPT_TRANSLATION_FORMAT.format(
            lang=target_language.value,
            original_item=original_claim,
            free_translation=resp,
        )
        resp = await client.text_generation(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.3,
            grammar={"type": "json", "value": _ClaimModel.model_json_schema()},
        )
    except Exception as e:
        logger.error(
            f"Failed to format claim gist and label {node_data} given {resp} due to {e}"
        )
        raise e

    try:
        translated_node_data = json.loads(resp)
        node_data = node_data.copy()
        node_data.update(translated_node_data)
    except Exception as e:
        logger.debug(
            f"Node data: {node_data}\n"
            f"Format prompt: {prompt}\n"
            f"Response: {resp}\n"
            f"Response type: {type(resp)}\n"
        )
        logger.error(f"Failed to update node {node_data} with {resp} due to {e}")
        raise e

    if "premises" in node_data and node_data["premises"]:
        try:
            translated_claim = _ClaimModel(**translated_node_data)
            original_premises = _PremisesModel(premises=node_data["premises"])
            prompt = _PROMPT_TRANSLATION_PREMISES.format(
                lang=target_language.value,
                original_claim=original_claim,
                translated_claim=translated_claim.model_dump_json(),
                original_premises="\n* ".join(original_premises.premises),
            )
            resp = await client.text_generation(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.4,
            )
        except Exception as e:
            logger.error(f"Failed to translate premises {node_data} due to {e}")
            raise e
        try:
            prompt = _PROMPT_TRANSLATION_FORMAT.format(
                lang=target_language.value,
                original_item=original_premises.model_dump_json(),
                free_translation=resp,
                schema=_PremisesModel.model_json_schema(),
            )
            resp = await client.text_generation(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.3,
                grammar={"type": "json", "value": _PremisesModel.model_json_schema()},
            )
        except Exception as e:
            logger.error(
                f"Failed to format premises {node_data} given {resp} due to {e}"
            )
            raise e
        try:
            node_data.update(json.loads(resp))
            # Make sure that premise list length is preserved!
            if not node_data["premises"]:
                node_data["premises"] = original_premises.premises
            elif len(node_data["premises"]) < len(original_premises.premises):
                node_data["premises"] += node_data["premises"][-1] * (
                    len(original_premises.premises) - len(node_data["premises"])
                )
            elif len(node_data["premises"]) > len(original_premises.premises):
                node_data["premises"] = node_data["premises"][
                    : len(original_premises.premises)
                ]
        except Exception as e:
            logger.error(
                f"Failed to update premises node {node_data} with {resp} due to {e}"
            )
            raise e
        logger.debug(f"Translation: {str(node_data)[:100]}...")

    return node_data


async def translate_argmap(source_argmap: nx.DiGraph, **kwargs):
    client_kwargs = {
        "token": kwargs["hf_token"],
        "headers": {"X-use-cache": "false"},
    }
    if "base_url" in kwargs:
        client_kwargs["base_url"] = kwargs["base_url"]
    else:
        client_kwargs["model"] = kwargs["model"]
    logger.debug(f"Initializing AsyncInferenceClient with {client_kwargs}")
    client = AsyncInferenceClient(**client_kwargs)

    target_language = getattr(Language, kwargs["target_language"])
    target_argmap = source_argmap.copy()

    translated_nodes = []

    async def translate_node(
        node,
        parent: str | None = None,
        translated_parent_data: dict | None = None,
        original_parent_data: dict | None = None,
    ):
        if node in translated_nodes:
            return
        original_node_data = target_argmap.nodes[node].copy()
        try:
            if parent is None:
                translated_node_data = await _translate_root(
                    original_node_data, target_language, client=client
                )
            else:
                translated_node_data = await _translate_reason(
                    original_node_data,
                    target_language,
                    client=client,
                )
            nx.set_node_attributes(target_argmap, {node: translated_node_data})
        except Exception as e:
            logger.error(f"Failed to translate node {original_node_data} due to {e}. Will keep original data.")

        translated_nodes.append(node)

        for child in target_argmap.predecessors(node):
            await translate_node(
                child,
                parent=node,
                translated_parent_data=translated_node_data,
                original_parent_data=original_node_data,
            )

        return

    for root in target_argmap.nodes:
        if not list(target_argmap.successors(root)):
            await translate_node(root)

    return target_argmap
