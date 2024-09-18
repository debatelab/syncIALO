import networkx as nx
from operator import itemgetter
import random
import uuid

from loguru import logger

import datasets

from syncialo.chains.argumentation import (
    IdentifyPremisesChain,
    GenerateProAndConChain,
    SelectMostSalientChain,
    Valences,
)

_ARGS_PER_PERSONA = 2


class DebateBuilder:

    def __init__(self, tags_universal, tags_per_cluster, model):
        self.tags_universal = tags_universal
        self.tags_per_cluster = tags_per_cluster
        self.model = model

        # build sub-chains
        self.identify_premises = IdentifyPremisesChain.build(model)
        self.chain_generate_pro_and_con = GenerateProAndConChain.build(model)
        self.select_most_salient = SelectMostSalientChain.build(model)

        # download and init persona datasets
        ds = datasets.load_dataset("proj-persona/PersonaHub", "reasoning", split="train")
        self.ds_personas = ds.select_columns(["input persona"])

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

        """

        depth = nx.shortest_path_length(tree, source=node_id, target=root_id)
        degree = degree_config[depth]

        logger.debug(f"Processing at depth {depth}")
        logger.debug(f"Degree = {degree}")
        logger.debug(f"Reason claim {tree.nodes[node_id]['claim']}")

        if not degree:
            return

        # identify premises of target node (i.e., node_id)
        if node_id != root_id:

            _, parent_id, data = next(iter(tree.out_edges(node_id, data=True)), (None, None, None))
            if parent_id is None:
                raise ValueError("Node %s has no parent node." % node_id)

            premises = tree.nodes[node_id].get('premises')
            if premises is None:
                premises = await self.identify_premises.ainvoke(
                    argument=tree.nodes[node_id]['claim'],
                    conclusion=tree.nodes[parent_id]['claim'],
                    valence=data['valence'],
                )
                tree.nodes[node_id]['premises'] = premises

        else:
            premises = [tree.nodes[node_id]['claim']]

        if not premises:
            logger.warning(f"No premises found for node: {tree.nodes[node_id]['claim']}. Skip building subtree.")
            return

        personas = ["A farmer from Ohio."] * degree  # TODO: sample from dataset
        batched_input = [
            {"premises": premises, "tags": tags, "persona": persona, "n": _ARGS_PER_PERSONA}
            for persona in personas
        ]

        # generate 2*n*degree arguments
        generated_args = await self.chain_generate_pro_and_con.abatch(batched_input)

        # select k most salient, mutually independent args
        all_generated_pros = [arg["new_pro"] for arg in generated_args if arg["new_pro"]]
        all_generated_cons = [arg["new_con"] for arg in generated_args if arg["new_con"]]

        salient_pros = self.select_most_salient.ainvoke(
            args=all_generated_pros,
            conclusion=tree.nodes[node_id]['claim'],
            valence=Valences.PRO,
            k=degree,
        )
        salient_cons = self.select_most_salient.ainvoke(
            args=all_generated_cons,
            conclusion=tree.nodes[node_id]['claim'],
            valence=Valences.CON,
            k=degree,
        )

        # add new nodes and edges to tree
        con_ids = []
        pro_ids = []
        for new_pro in salient_pros:
            uid = str(uuid.uuid4())
            tree.add_node(
                uid,
                claim=new_pro["claim"],
                label=new_pro["label"],
            )
            tree.add_edge(
                uid,
                node_id,
                valence=Valences.PRO,
                target_idx=new_pro["target_idx"])
            pro_ids.append(uid)
        for new_con in salient_cons:
            uid = str(uuid.uuid4())
            tree.add_node(
                uid,
                claim=new_con["claim"],
                label=new_con["label"],
            )
            tree.add_edge(
                uid,
                node_id,
                valence=Valences.CON,
                target_idx=new_con["target_idx"])
            con_ids.append(uid)

        # recursion
        for pro_id in pro_ids:
            await self.build_subtree(
                node_id=pro_id,
                root_id=root_id,
                tree=tree,
                degree_config=degree_config,
                tags=tags,
            )

        for con_id in con_ids:
            await self.build_subtree(
                node_id=con_id,
                root_id=root_id,
                tree=tree,
                degree_config=degree_config,
                tags=tags,
            )




        ## BREAK ##








        # rank by plausibility
        if n_prem > 1:
            ranking = await self.rank_by_plausibility.ainvoke(
                premises=premises,
                tags=tags,
                persona=None,
            )
        else:
            ranking = list(range(n_prem))


        # generate and add pros
        pro_ids = []
        # TODO: needs to be discussed
        target_idx = ranking[0]  # support most plausible
        pros = await self.gen_supporting_argument.ainvoke(
            premises=premises,
            ranking=ranking,
            tags=random.sample(self.tags_universal, k=self.tags_per_cluster),  # resample tags for more diversity
            persona=None,
        )

        for claim in pros:
            uid = str(uuid.uuid4())
            tree.add_node(uid, claim=claim, target_proposition=premises[target_idx])
            tree.add_edge(uid, node_id, valence=Valences.PRO, target_idx=target_idx)
            pro_ids.append(uid)

        # generate and add cons
        con_ids = []
        # TODO: needs to be discussed
        target_idx = ranking[-1]  # attack least plausible only
        cons = await self.gen_attacking_argument.ainvoke(
            premises=premises,
            ranking=ranking,
            tags=random.sample(self.tags_universal, k=self.tags_per_cluster),  # resample tags for more diversity
            persona=None,
        )

        # TODO: add peer-review and revise step for cons
        # TODO: Check for duplicates in entire tree/DAG and match arguments

        for claim in cons:
            uid = str(uuid.uuid4())
            tree.add_node(uid, claim=claim, target_proposition=premises[target_idx])
            tree.add_edge(uid, node_id, valence=Valences.CON, target_idx=target_idx)
            con_ids.append(uid)

        # recursion
        for pro_id in pro_ids:
            await self.build_subtree(
                node_id=pro_id,
                root_id=root_id,
                tree=tree,
                degree_config=degree_config,
                tags=tags,
            )

        for con_id in con_ids:
            await self.build_subtree(
                node_id=con_id,
                root_id=root_id,
                tree=tree,
                degree_config=degree_config,
                tags=tags,
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
                data['valence']
            )

    root_id = next(n for n in tree.nodes if len(tree.out_edges(n)) == 0)
    counter = "1."

    add_node(root_id, counter)

    return lines
