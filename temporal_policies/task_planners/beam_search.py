import ast
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from torch import Tensor
from helm.common.authentication import Authentication
from configs.base_config import LMConfig
from temporal_policies import envs
from temporal_policies.envs.base import Primitive
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.task_planners.lm_data_structures import (
    CurrentExample,
    InContextExample,
)
from temporal_policies.task_planners.lm_utils import (
    APIType,
    generate_lm_response,
    get_examples_from_json_dir,
)
from temporal_policies.envs.pybullet.table import predicates

from temporal_policies.evaluation.utils import (
    get_goal_props_instantiated,
    get_object_relationships,
    get_possible_props,
    get_task_plan_primitives_instantiated,
)
from temporal_policies.task_planners.task_plans import (
    get_action_scores_from_lm,
    get_next_actions_from_lm,
)


# Node = namedtuple('Node', ['optimized_action_distribution', 'action_primitive', 'depth'])
class Node:
    def __init__(
        self,
        parent: "Node",
        action_primitive: Primitive,
        optimized_action_parameters: Optional[List[Tensor]] = None,
        optimized_action_distribution: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        self.parent = parent
        self.action_primitive = action_primitive
        self._action_sequence_score: Optional[Tensor] = None
        self._env = None
        # TODO(klin): unclear if the parameters are just the mode of the distribution?
        self._optimized_action_parameters = optimized_action_parameters
        self._optimized_action_distribution = optimized_action_distribution
        self._optimized_final_state: Optional[Tensor] = None
        self._optimized_object_relationships: Optional[List[str]] = None
        # computed by rolling out dynamics on the highest product of Q value
        # continuous action parameters

    @property
    def optimized_object_relationships(self):
        if self._optimized_object_relationships is None:
            raise ValueError(
                "Optimized object relationships (i.e. relationships from \
                rolling out dynamics on the highest product of Q value \
                continuous action parameters"
            )
        return self._optimized_action_distribution

    @property
    def optimized_action_distribution(self):
        if self._optimized_action_distribution is None:
            raise ValueError("Optimized action distribution not set")
        return self._optimized_action_distribution

    @optimized_action_distribution.setter
    def optimized_action_distribution(self, value: Tuple[Tensor, Tensor]):
        self._optimized_action_distribution = value

    @property
    def sequence_length(self):
        if self.parent is None:
            return 1
        return self.parent.sequence_length + 1

    @property
    def action_sequence_score_lm(self):
        if self._action_sequence_score is None:
            raise ValueError("Optimized action distribution not set")
        return self._action_sequence_score

    @action_sequence_score_lm.setter
    def action_sequence_score_lm(self, value: Tensor):
        self._action_sequence_score_lm = value

    @property
    def action_sequence_optimized_q_product(self):
        raise NotImplementedError("Cost full sequence Q not implemented")

    @property
    def action_sequence_overall_score(self):
        return (
            self.action_sequence_optimized_q_product / self.sequence_length
        ) * self.cost_lm  # unclear how to trade-off the weight of the two costs
        # one way is to divide the cost of the full sequence Q by the number of primitives in the sequence

    @property
    def action_sequence(self) -> List[str]:
        action_sequence_lst: List[str] = [str(self.action_primitive).lower()]
        if self.parent is None:
            return action_sequence_lst
        action_sequence_lst.extend(self.parent.action_sequence)
        return action_sequence_lst

    @property
    def object_relationships_sequence(self) -> List[str]:
        object_relationships_sequence: List[str] = [str(self.action_primitive).lower()]
        if self.parent is None:
            return action_sequence_lst
        action_sequence_lst.extend(self.parent.action_sequence)
        return action_sequence_lst


def expand_node(current_node: Node) -> List[Node]:
    new_nodes: List[Node] = []
    # another way is to first
    # current node should store the current object relationships I think?
    # mayhbe each node should contain the env??
    # or, have a little object inside the node that stores info required to
    # compute the next node?
    actions, lm_cache = get_next_actions_from_lm(
        current_node.instruction,
        goal,
        objects,
        object_relationships,
        all_prior_object_relationships,
        all_executed_actions,
    )  # could wrap this function in the 'expand_node' function for abstractions ...
    # save lm_cache, actions is List[str]
    # get the primitives from the actions list

    env_lst = [env] * len(actions)
    potential_actions: List[table_primitives.Primitive] = [
        env.get_primitive_info(action, env) for (action, env) in zip(actions, env_lst)
    ]

    new_nodes = []
    for potential_action in potential_actions:
        new_node = Node(node, potential_action)
        new_nodes.append(new_node)

    # optimize the Q values here using CEM

    # potentially prune off completely infeasible actions first, so we don't need to score as meany
    return new_nodes


def beam_search(
    start_node: Node,
    expand_function,
    expand_optimized_function,
    success_checker_fn: Callable,
    beam_size: int = 3,
    max_depth: int = 8,
):
    """
    :param start_node: Node to start the search from
    """
    beam: List[Node] = [start_node]

    # Iterate over the depths
    for depth in range(max_depth + 1):
        # Initialize the next beam
        next_beam: List[Node] = []

        # Iterate over the nodes in the current beam
        for node in beam:
            # If the depth is less than the maximum depth, use the discrete expansion function
            if node.depth < max_depth:
                # Expand the node to generate new nodes
                new_nodes = expand_node(node)
                # get new nodes by taking current node and getting the top K actions LM generated
                # Add the new nodes to the next beam
                next_beam.extend(new_nodes)

        # extract the full action sequences for each node in next_beam and then score them
        action_sequences = [node.action_sequence for node in next_beam]
        # TODO(klin)
        action_sequence_lm_scores = get_action_sequence_scores_from_lm(instruction)

        for i, node in enumerate(next_beam):
            node.action_sequence_score_lm = action_sequence_lm_scores[i]

        # Sort the next beam in ascending order of optimization
        next_beam.sort(key=lambda x: x.action_sequence_overall_score)

        # Trim the next beam to the specified width
        next_beam = next_beam[:beam_size]

        # Set the current beam to the next beam
        beam = next_beam

    # Return the set of nodes in the final beam
    return beam


# # Example usage
# start_node = Node(optimized_action_distribution=0, action_primitive='start', depth=0)
# beam_search(start_node, expand_function, expand_optimized_function, beam_width=5, max_depth=3)
# In this modified version, the expand_function generates new nodes with an incremented depth by executing the action primitive of the input node. The expand_optimized_function generates a new node with an optimized action distribution by optimizing the action distribution up to the input node's action primitive.
