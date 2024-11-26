import enum
import json

from huggingface_hub import InferenceClient
from loguru import logger
import networkx as nx
from pydantic import BaseModel
import tenacity

from syncialo.chains.argumentation import Valence


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
    "I'm translating a debate from English to {lang}. "
    "Can you please help me and translate the following claim: 'f{original_claim}'. "
    "Stick to the json format without translating the keys."
)

_PROMPT_TRANSLATION_REASON = (
        "I ask you to assist me in translating a debate from English to {lang}. "
        "Previously, I have translated: "
        "'f{original_parent}' as 'f{translated_parent}'.\n"
        "Can you please translate the next "
        "claim, which {phis} the previous one mentioned above, to {lang}: 'f{original_claim}'. "
        "Stick to the json format without translating the keys."
)

_PROMPT_TRANSLATION_PREMISES = (
        "I ask you to assist me in translating a debate from English to {lang}. "
        "I have translated the the gist of an argument: "
        "'f{original_claim}' as 'f{translated_claim}'.\n"
        "Next, I need to translate the arguments premises. "
        "Can you please help me and translate the following list of sentences, "
        "all of which are premises of the argument: "
        "'f{original_premises}'. "
        "Provide your translations as a list of {lang} sentences, "
        "sticking to the json format (without translating the keys)."
)


@tenacity.retry(wait=tenacity.wait_random_exponential(multiplier=1, max=60))
def _translate_root(node_data: dict, target_language: Language, client: InferenceClient) -> dict:
    claim = _ClaimModel(claim=node_data["claim"], label=node_data["label"])
    prompt = _PROMPT_TRANSLATION_ROOT.format(lang=target_language.value, original_claim=claim.model_dump_json())
    resp = client.text_generation(
        prompt=prompt,
        max_new_tokens=250,
        temperature=0.4,
        seed=42,
        grammar={"type": "json", "value": _ClaimModel.model_json_schema()},
    )
    node_data = node_data.copy()
    node_data.update(json.loads(resp))
    return node_data


@tenacity.retry(wait=tenacity.wait_random_exponential(multiplier=1, max=60))
def _translate_reason(
    node_data: dict,
    translated_parent: dict,
    original_parent: dict,
    valence: Valence,
    target_language: Language,
    client: InferenceClient,
) -> dict:
    original_claim = _ClaimModel(claim=node_data["claim"], label=node_data["label"])
    translated_parent_claim = _ClaimModel(claim=translated_parent["claim"], label=translated_parent["label"])
    original_parent_claim = _ClaimModel(claim=original_parent["claim"], label=original_parent["label"])
    phis = "supports" if valence == Valence.PRO else "attacks"
    prompt = _PROMPT_TRANSLATION_REASON.format(
        lang=target_language.value,
        original_parent=original_parent_claim.model_dump_json(),
        translated_parent=translated_parent_claim.model_dump_json(),
        phis=phis,
        original_claim=original_claim.model_dump_json(),
    )
    resp = client.text_generation(
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.4,
        seed=42,
        grammar={"type": "json", "value": _ClaimModel.model_json_schema()},
    )
    translated_node_data = json.loads(resp)    
    node_data = node_data.copy()
    node_data.update(translated_node_data)

    if "premises" in node_data and node_data["premises"]:
        translated_claim = _ClaimModel(**translated_node_data)
        original_premises = _PremisesModel(premises=node_data["premises"])
        prompt = _PROMPT_TRANSLATION_PREMISES.format(
            lang=target_language.value,
            original_claim=original_claim.model_dump_json(),
            translated_claim=translated_claim.model_dump_json(),
            original_premises=original_premises.model_dump_json(),
        )
        resp = client.text_generation(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.4,
            seed=42,
            grammar={"type": "json", "value": _PremisesModel.model_json_schema()},
        )
        node_data.update(json.loads(resp))

    return node_data


def translate_argmap(source_argmap: nx.DiGraph, **kwargs):

    client_kwargs = {
        "hf_token": kwargs["hf_token"],
    }
    if "base_url" in kwargs:
        client_kwargs["base_url"] = kwargs["base_url"]
    else:
        client_kwargs["model"] = kwargs["model"]
    client = InferenceClient(**client_kwargs)

    target_language = kwargs["target_language"]
    target_argmap = source_argmap.copy()

    translated_nodes = []

    def translate_node(
        node,
        parent: str | None = None,
        translated_parent_data: dict | None = None,
        original_parent_data: dict | None = None,
    ):
        if node in translated_nodes:
            return
        original_node_data = target_argmap.nodes[node].copy()
        if parent is None:
            translated_node_data = _translate_root(original_node_data, target_language, client=client)
        else:
            valence = Valence(
                target_argmap.edges[node, parent]["valence"]
            )
            translated_node_data = _translate_reason(
                original_node_data,
                translated_parent_data,
                original_parent_data,
                valence,
                target_language,
                client=client,
            )
        translated_nodes.append(node)
        nx.set_node_attributes(target_argmap, {node: translated_node_data})

        for child in target_argmap.predecessors(node):
            translate_node(
                child,
                parent=node,
                translated_parent_data=translated_node_data,
                original_parent_data=original_node_data,
            )

        return

    for root in target_argmap.nodes:
        if not list(target_argmap.successors(root)):
            translate_node(root)

    return target_argmap
