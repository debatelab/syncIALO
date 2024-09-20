import networkx as nx
import random
import uuid

from loguru import logger

import datasets

from syncialo.chains.argumentation import (
    IdentifyPremisesChain,
    GenerateProAndConChain,
    SelectMostSalientChain,
    ArgumentModel,
    Valences,
)

_ARGS_PER_PERSONA = 2
_PERSONAS_DATASET = dict(
    path="proj-persona/PersonaHub",
    name="reasoning",
    split="train"
)
_TAGS_PER_CLUSTER = 8


class DebateBuilder:

    def __init__(self, model, **kwargs):
        self.model = model

        if "tags_universal" not in kwargs:
            raise ValueError("Argument 'tags_universal' is required.")
        self.tags_universal = kwargs["tags_universal"]
        self.tags_per_cluster = kwargs.get("tags_per_cluster", _TAGS_PER_CLUSTER)
        self.tags_eval = kwargs.get("tags_eval", None)
        self.tags_test = kwargs.get("tags_test", None)
        self.split = kwargs.get("split", "train")
        if self.split not in ["train", "eval", "test"]:
            raise ValueError("Argument 'split' must be one of 'train', 'eval', 'test'.")
        if self.split == "eval" and not self.tags_eval:
            raise ValueError("Argument 'tags_eval' is required for split 'eval'.")
        if self.split == "test" and not self.tags_test:
            raise ValueError("Argument 'tags_test' is required for split 'test'.")

        # build sub-chains
        self.chain_identify_premises = IdentifyPremisesChain.build(model)
        self.chain_generate_pro_and_con = GenerateProAndConChain.build(model)
        self.chain_select_most_salient = SelectMostSalientChain.build(model)

        # download and init persona datasets
        ds = datasets.load_dataset(**_PERSONAS_DATASET)
        self.ds_personas = ds.select_columns(["input persona"])

    async def identify_premises(self, node_id: str, root_id: str, tree: nx.DiGraph) -> list[str]:
        """
        checks if premises of node_id have already been identified,
        otherwise calls LLM-chain to do so, caches and returns result
        """
        if node_id == root_id:
            return [tree.nodes[node_id]['claim']]

        _, parent_id, data = next(iter(tree.out_edges(node_id, data=True)), (None, None, None))
        if parent_id is None:
            raise ValueError("Node %s has no parent node." % node_id)

        # look-up premises
        premises = tree.nodes[node_id].get('premises')

        if premises is None:
            premises = await self.chain_identify_premises.ainvoke({
                "argument": tree.nodes[node_id]['claim'],
                "conclusion": tree.nodes[parent_id]['claim'],
                "valence": data['valence'],
            })
            # cache premises as node attribute in tree
            tree.nodes[node_id]['premises'] = premises

        return premises

    @logger.catch
    async def build_subtree(
        self,
        node_id: str,
        root_id: str,
        tree: nx.DiGraph,
        degree_config: list,
        tags: list,
        topic: str,
    ):
        """
        builds the subtree under node_id and adds it to tree

        args:

            node_id:       root of subtree to build
            root_id:       root of entire argmap
            tree:          entire argmap
            degree_config: list that details number of attacks / supports in function of depth
            tags:          tags for the particular debate currently built
        """

        depth = nx.shortest_path_length(tree, source=node_id, target=root_id)
        degree = degree_config[depth]  # number if pros / cons to generate
        logger.debug(f"Processing at depth {depth}")
        logger.debug(f"Degree = {degree}")
        logger.debug(f"Reason claim {tree.nodes[node_id]['claim']}")
        if not degree:
            return

        persona_idxs = random.sample(range(len(self.ds_personas)), k=degree)
        personas: list[str] = self.ds_personas.select(persona_idxs)["input persona"]

        premises = await self.identify_premises(node_id, root_id, tree)
        if not premises:
            logger.warning(f"No premises found for node: {tree.nodes[node_id]['claim']}. Skip building subtree.")
            return

        batched_input = [
            {
                "premises": premises,
                "tags": tags,
                "tags_universal": self.tags_universal,
                "tags_per_cluster": self.tags_per_cluster,
                "persona": persona,
                "n": _ARGS_PER_PERSONA
            }
            for persona in personas
        ]

        # generate 2*n*degree arguments
        batched_generated_args = await self.chain_generate_pro_and_con.abatch(batched_input)
        all_generated_pros = [arg for gen_args in batched_generated_args for arg in gen_args["new_pros"]]
        all_generated_cons = [arg for gen_args in batched_generated_args for arg in gen_args["new_cons"]]

        # select k most salient, mutually independent args
        salient_pros: list[ArgumentModel] = await self.chain_select_most_salient.ainvoke({
            "args": all_generated_pros,
            "conclusion": tree.nodes[node_id]['claim'],
            "valence": Valences.PRO,
            "k": degree,
        })
        salient_cons: list[ArgumentModel] = await self.chain_select_most_salient.ainvoke({
            "args": all_generated_cons,
            "conclusion": tree.nodes[node_id]['claim'],
            "valence": Valences.CON,
            "k": degree,
        })

        # add new nodes and edges to tree
        con_ids = []
        pro_ids = []
        for new_pro in salient_pros:
            uid = str(uuid.uuid4())
            tree.add_node(
                uid,
                claim=new_pro.claim,
                label=new_pro.label,
            )
            tree.add_edge(
                uid,
                node_id,
                valence=Valences.PRO,
                target_idx=new_pro.target_idx)
            pro_ids.append(uid)
        for new_con in salient_cons:
            uid = str(uuid.uuid4())
            tree.add_node(
                uid,
                claim=new_con.claim,
                label=new_con.label,
            )
            tree.add_edge(
                uid,
                node_id,
                valence=Valences.CON,
                target_idx=new_con.target_idx)
            con_ids.append(uid)

        # recursion
        for pro_id in pro_ids:
            await self.build_subtree(
                node_id=pro_id,
                root_id=root_id,
                tree=tree,
                degree_config=degree_config,
                tags=tags,
                topic=topic,
            )

        for con_id in con_ids:
            await self.build_subtree(
                node_id=con_id,
                root_id=root_id,
                tree=tree,
                degree_config=degree_config,
                tags=tags,
                topic=topic,
            )

    async def build_debate(
        self,
        root_claim: str,
        topic: str,
        tag_cluster,
        degree_config,
    ) -> nx.DiGraph:

        tree = nx.DiGraph()

        root_id = str(uuid.uuid4())
        tree.add_node(
            root_id,
            claim=root_claim
        )

        await self.build_subtree(
            node_id=root_id,
            root_id=root_id,
            tree=tree,
            degree_config=degree_config,
            tags=tag_cluster,
            topic=topic,
        )

        # TODO: Check for duplicate labels and revise/specify labels if necessary
        # TODO: Check for duplicate arguments in entire tree/DAG and match arguments

        return tree


def to_kialo(tree, topic=""):

    lines = []
    lines.append(f"Discussion Title: {topic}")
    lines.append("")

    def add_node(target, counter, val=None):

        if val is None:
            sym = " "
        else:
            sym = " PRO: " if val == Valences.PRO else " CON: "

        line = counter + sym + tree.nodes[target]["claim"]
        lines.append(line)

        i = 0
        for source, _, data in tree.in_edges(target, data=True):
            i += 1
            add_node(
                source,
                counter+f"{i}.",
                data["valence"].value
            )

    root_id = next(n for n in tree.nodes if len(tree.out_edges(n)) == 0)
    counter = "1."

    add_node(root_id, counter)

    return lines
