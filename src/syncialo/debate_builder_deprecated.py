import networkx as nx
import random
import uuid

from loguru import logger

from queries import identify_premises, rank_by_plausibility, supporting_argument, attacking_argument, PRO, CON


class DebateBuilder:

    def __init__(self, tags_universal, tags_per_cluster, model):
        self.tags_universal = tags_universal
        self.tags_per_cluster = tags_per_cluster
        self.model = model

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

        # get premises
        if node_id != root_id:

            _, parent_id, data = next(iter(tree.out_edges(node_id, data=True)), (None, None, None))
            if parent_id is None:
                raise ValueError("Node %s has no parent node." % node_id)

            premises = tree.nodes[node_id].get('premises')
            if premises is None:
                premises = await identify_premises(
                    model=self.model,
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

        n_prem = len(premises)

        # rank by plausibility
        if n_prem > 1:
            ranking = await rank_by_plausibility(
                model=self.model,
                premises=premises,
                tags=tags,
                decoder="beam",
                n=2,
            )
            ranking = ranking[0]
        else:
            ranking = list(range(n_prem))

        n = degree  # number of pros and cons to generate

        # generate and add pros
        pro_ids = []
        target_idx = ranking[0]  # support most plausible
        pros = await supporting_argument(
            model=self.model,
            premises=premises,
            target_idx=target_idx,
            tags=random.sample(self.tags_universal, k=self.tags_per_cluster),  # resample tags for more diversity
            n=n,
            decoder="sample",
            temperature=.6,    
        )

        for claim in pros:
            uid = str(uuid.uuid4())        
            tree.add_node(uid, claim=claim)
            tree.add_edge(uid, node_id, valence=PRO, target_idx=target_idx)
            pro_ids.append(uid)

        # generate and add cons
        con_ids = []
        target_idx = ranking[-1]  # attack most unplausible
        cons = await attacking_argument(
            model=self.model,
            premises=premises,
            target_idx=target_idx,
            tags=random.sample(self.tags_universal, k=self.tags_per_cluster),  # resample tags for more diversity
            n=n,
            decoder="sample",
            temperature=.6,
        )

        for claim in cons:
            uid = str(uuid.uuid4())        
            tree.add_node(uid, claim=claim)
            tree.add_edge(uid, node_id, valence=CON, target_idx=target_idx)
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
        
        return tree
    
    
def to_kialo(tree, topic=""):

    lines = []
    lines.append(f"Discussion Title: {topic}")
    lines.append("")

    def add_node(target, counter, val=None):

        if val is None:
            sym = " "
        else:
            sym = " PRO: " if val == PRO else " CON: "

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

    root_id = next(n for n in tree.nodes if len(tree.out_edges(n))==0)
    counter = "1."

    add_node(root_id, counter)
    
    return lines
    
    