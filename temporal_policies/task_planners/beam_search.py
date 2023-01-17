from functools import cached_property
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from termcolor import colored
from scripts.eval.eval_saycan import format_saycan_scoring_table
from torch import Tensor

# helm should be optional
from helm.common.authentication import Authentication
import tabulate

tabulate.PRESERVE_WHITESPACE = True
from tabulate import tabulate

from temporal_policies import envs, planners
from temporal_policies.envs.base import Primitive
from temporal_policies.task_planners.lm_utils import (
    save_lm_cache,
)
from temporal_policies.envs.pybullet.table import predicates, primitives

from temporal_policies.evaluation.utils import (
    get_object_relationships,
    instantiate_task_plan_primitives,
    is_satisfy_goal_props,
)
from temporal_policies.task_planners.task_plans import (
    get_action_scores_from_lm,
    get_next_actions_from_lm,
)

from temporal_policies import agents, envs, planners


class Node:
    def __init__(
        self,
        parent: Optional["Node"] = None,
        action_primitive: Optional[Primitive] = None,
        geometric_state: Optional[np.ndarray] = None,
        motion_planner: Optional[planners.Planner] = None,
        env: Optional[envs.Env] = None,
        available_predicates: Optional[List[predicates.Predicate]] = None,
    ):
        self.action_primitive = action_primitive
        self.parent = parent
        self._geometric_state = geometric_state
        self._motion_planner = motion_planner
        self._env = env
        self._available_predicates = available_predicates
        self._action_sequence_score: Optional[float] = None

    @property
    def env(self) -> envs.Env:
        if self.parent is None:
            return self._env
        return self.parent.env

    @property
    def available_predicates(self) -> List[predicates.Predicate]:
        if self.parent is None:
            return self._available_predicates
        return self.parent.available_predicates

    @property
    def motion_planner(self) -> planners.Planner:
        if self.parent is None:
            return self._motion_planner
        return self.parent.motion_planner

    @property
    def sequence_length(self):
        if self.parent is None:
            return 1
        return self.parent.sequence_length + 1

    @property
    def action_sequence_score_lm(self):
        if self._action_sequence_score_lm is None:
            raise ValueError("action_sequence_score_lm not set")
        return self._action_sequence_score_lm

    @action_sequence_score_lm.setter
    def action_sequence_score_lm(self, value: float):
        self._action_sequence_score_lm = value

    @property
    def action_sequence_q_product_post_optimization(self):
        return self.motion_plan_post_optimization.values.prod()

    @property
    def root_node_geometric_state(self) -> np.ndarray:
        if self.parent is None:
            return self._geometric_state
        return self.parent.root_node_geometric_state

    @property
    def root_node_object_relationships(self) -> List[str]:
        assert (
            self.parent is None
        ), "root_node_object_relationships only defined for root node"
        # parse the initial state to get the object relationships
        return get_object_relationships(
            self.root_node_geometric_state,
            self.env.objects,
            self.available_predicates,
            use_hand_state=False,
        )

    @property
    def beam(self) -> List["Node"]:
        """List of nodes in the beam; used for recording and visualization purposes"""
        return self._beam

    @beam.setter
    def beam(self, value: List["Node"]):
        self._beam = value

    @property
    def custom_recording_text(self) -> str:
        """Custom text to be printed during beam search"""
        if not hasattr(self, "_custom_recording_text"):
            return ""
        return self._custom_recording_text

    @custom_recording_text.setter
    def custom_recording_text(self, value: str):
        self._custom_recording_text = value

    @cached_property
    def custom_recording_text_sequence(self) -> List[str]:
        """Custom text to be printed during beam search"""
        if self.parent is None:
            return []
        return self.parent.custom_recording_text_sequence + [self.custom_recording_text]

    @property
    def action_skeleton_as_strings(self) -> Optional[List[str]]:
        if self.action_primitive is None:
            return []
        action_skeleton_lst: List[str] = self.parent.action_skeleton_as_strings
        action_skeleton_lst.append(str(self.action_primitive).lower())
        return action_skeleton_lst

    @property
    def action_skeleton_as_primitives(self) -> List[primitives.Primitive]:
        if self.action_primitive is None:
            return []
        action_skeleton_lst: List[str] = self.parent.action_skeleton_as_primitives
        action_skeleton_lst.append(self.action_primitive)
        return action_skeleton_lst

    @cached_property
    def motion_plan_post_optimization(self) -> planners.PlanningResult:
        """Full motion plan after optimization"""
        return self.motion_planner.plan(
            self.root_node_geometric_state,
            instantiate_task_plan_primitives(self.action_skeleton_as_strings, self.env),
        )

    @cached_property
    def object_relationships_sequence_post_optimization(self) -> List[List[str]]:
        """All object relationships in the sequence of actions after optimization"""
        # for each state in the motion plan, get the object relationships
        object_relationships_sequence: List[List[str]] = []
        for state in self.motion_plan_post_optimization.states:
            object_relationships = get_object_relationships(
                state,
                self.env.objects,
                self.available_predicates,
                use_hand_state=False,
            )
            object_relationships_sequence.append(
                [str(obj_rel) for obj_rel in object_relationships]
            )
        return object_relationships_sequence

    @property
    def potential_actions_from_lm(self) -> List[str]:
        """Next possible actions (according to the language model)"""
        if self._potential_actions_from_lm is None:
            raise ValueError("Potential actions from LM not set")
        return self._potential_actions_from_lm

    @potential_actions_from_lm.setter
    def potential_actions_from_lm(self, potential_actions: List[str]):
        self._potential_actions_from_lm = potential_actions


class SearchProblem:
    def __init__(self) -> None:
        pass

    def is_end(self, node: Node) -> bool:
        raise NotImplementedError

    def get_successors(self, node: Node) -> List[Node]:
        raise NotImplementedError

    def get_node_scores(self, nodes: List[Node]) -> List[float]:
        raise NotImplementedError


class BeamSearchProblem(SearchProblem):
    def __init__(
        self,
        instruction: str,
        goal_props: List[str],
        initial_geometric_state: Tensor,
        planner: planners.Planner,
        env: envs.Env,
        available_predicates: List[str],
        prop_testing_objs: List[str],
        goal_props_callable: Callable[[List[str]], bool],
        pddl_domain_file: str,
        pddl_problem_file: str,
        examples: List[Dict[str, Any]],
        lm_cfg: Dict[str, Any],
        auth: Optional[Authentication] = None,
        lm_cache: Optional[Dict[str, str]] = None,
        lm_cache_file: Optional[str] = None,
    ):
        # TODO(klin) remove when instruction is inside env
        self.instruction: str = instruction
        self.goal_props: List[str] = goal_props
        self.planner: planners.Planner = planner
        self.env = env
        self.available_predicates = available_predicates  # TODO(klin)
        self.prop_testing_objs = (
            prop_testing_objs  # TODO(klin): potential refactor to be in env
        )
        self.goal_props_callable = goal_props_callable
        self.pddl_domain_file = pddl_domain_file  # TODO(klin); this is used for testing task plans' feasibility
        self.pddl_problem_file = pddl_problem_file
        # these belong in some LM object
        self.examples = examples
        self.lm_cfg = lm_cfg
        self.auth = auth
        self.lm_cache = lm_cache
        self.lm_cache_file = lm_cache_file

        # though maybe this can be hardcoded
        # since it differs for each type of LM call
        # self.custom_in_context_example_robot_prompt,
        # self.custom_in_context_example_robot_format,
        # self.custom_robot_prompt,
        # self.custom_robot_action_skeleton_format,

        self.start_node = Node(
            geometric_state=initial_geometric_state,
            motion_planner=planner,
            env=env,
            available_predicates=available_predicates,
        )

    def is_end(self, node: Node) -> bool:
        """Returns True if the node is a goal state."""
        print(
            colored(
                f"Checking if node is end: {node.action_skeleton_as_strings}", "green"
            )
        )
        print(f"Action skeleton values: {node.motion_plan_post_optimization.values}")
        print(
            f"Object relationships: {node.object_relationships_sequence_post_optimization[-1]}"
        )
        if is_satisfy_goal_props(
            self.goal_props_callable,
            self.prop_testing_objs,
            node.motion_plan_post_optimization.states[-1],
            use_hand_state=False,
        ):
            print(colored(f"Success: {node.action_skeleton_as_strings}", "green"))
            # object relationships
            print(
                f"Object relationships: {node.object_relationships_sequence_post_optimization[-1]}"
            )
            return True
        return False

    def get_node_scores(self, nodes: List[Node]) -> List[float]:
        """
        For a given list of nodes, returns a list of scores
        where score = LM score * value function score.
        """
        best_values = [
            node.action_sequence_q_product_post_optimization for node in nodes
        ]

        # extract the full action sequences for each node in beam and then score them
        action_skeletons = [node.action_skeleton_as_strings for node in nodes]
        # the first N - 1 actions in the action sequence should be the same for all nodes
        # valid implementation of the assert
        for action_skeleton in action_skeletons[1:]:
            assert (
                action_skeletons[0][:-1] == action_skeleton[:-1]
            ), "Action skeletons are not all the same except for the last action"

        # object relationships from state sequence (includes shouldn't include "prior" object relationship
        # only include predicted object relationships

        # TODO(klin): skip scoring some of the action sequences if we know the Q product is below some threshold
        # TODO(klin): format this function so that it makes sense
        # what data does action (sequence) scoring need and what data does action (sequence) generation need?
        # both need everything so far: the main difference is formatting (with some model selection quirks)
        # with beam search, we can also include the final observation after the action being considered in the prompt
        potential_actions_str = [str(node.action_primitive).lower() for node in nodes]
        lm_action_scores, lm_cache = get_action_scores_from_lm(
            self.instruction,
            potential_actions_str,
            self.goal_props,
            list(self.env.objects.keys()),
            nodes[0].object_relationships_sequence_post_optimization[-1][:-1]
            if nodes[0].parent is not None
            else nodes[0].root_node_object_relationships,
            nodes[0].object_relationships_sequence_post_optimization[:-1]
            if nodes[0].parent is not None
            else [nodes[0].root_node_object_relationships],
            action_skeletons[0][:-1],
            self.pddl_domain_file,
            self.pddl_problem_file,
            examples=self.examples,
            lm_cfg=self.lm_cfg,
            auth=self.auth,
            lm_cache=self.lm_cache,
            lm_cache_file=self.lm_cache_file,
        )
        self.lm_cache = lm_cache
        save_lm_cache(pathlib.Path(self.lm_cache_file), lm_cache)

        for i in range(len(nodes)):
            nodes[i].action_sequence_score_lm = lm_action_scores[i]

        return [best_values[i] * lm_action_scores[i] for i in range(len(nodes))]

    def get_successors(self, node: Node, num_successors: int = 5) -> List[Node]:
        """Returns num_successors successors of a given node."""
        new_nodes: List[Node] = []
        # TODO(klin) should be able to configure custom things
        actions, lm_cache = get_next_actions_from_lm(
            self.instruction,
            self.goal_props,
            list(self.env.objects.keys()),
            node.object_relationships_sequence_post_optimization[-1]
            if node.parent is not None
            else node.root_node_object_relationships,
            node.object_relationships_sequence_post_optimization
            if node.parent is not None
            else [node.root_node_object_relationships],
            node.action_skeleton_as_strings,
            self.pddl_domain_file,
            self.pddl_problem_file,
            examples=self.examples,
            custom_in_context_example_robot_prompt="Top robot action sequence: ",
            custom_in_context_example_robot_format="python_list",
            custom_robot_prompt=f"Top {num_successors} next actions (python list): ",
            custom_robot_action_sequence_format="python_list",
            lm_cfg=self.lm_cfg,
            auth=self.auth,
            lm_cache=self.lm_cache,
        )
        self.lm_cache = lm_cache
        save_lm_cache(pathlib.Path(self.lm_cache_file), lm_cache)

        env_lst = [self.env] * len(actions)
        potential_action_primitives: List[primitives.Primitive] = [
            env.get_primitive_info(action, env)
            for (action, env) in zip(actions, env_lst)
        ]

        new_nodes = []
        for action_primitive in potential_action_primitives:
            new_node = Node(
                parent=node,
                action_primitive=action_primitive,
            )
            new_nodes.append(new_node)

        return new_nodes


class BeamSearchAlgorithm:
    def __init__(
        self,
        max_beam_size: int = 4,
        max_depth: int = 9,
        num_successors_per_node: int = 4,
        verbose: bool = False,
    ):
        assert max_beam_size == 1, "Current prompts only work with beam size 1"
        self.verbose = verbose
        self.max_beam_size = max_beam_size
        self.max_depth = max_depth
        self.num_successors_per_node = num_successors_per_node

    def filter_beam(self, beam: List, scores: List[float], max_beam_size: int) -> List:
        # sort beam according to scores: descending order
        beam = [x for _, x in sorted(zip(scores, beam), key=lambda pair: -pair[0])]

        # Trim the next beam to the specified width
        beam = beam[:max_beam_size]
        return beam

    def solve(
        self,
        problem: BeamSearchProblem,
        visualize: bool = False,
    ) -> List[Node]:
        """
        Solves a given problem using beam search.
        """
        beam: List[Node] = [problem.start_node]

        # Iterate over the depths
        for search_depth in range(self.max_depth + 1):
            # Initialize the next beam
            next_beam: List[Node] = []

            # Iterate over the nodes in the current beam
            for node in beam:
                if search_depth < self.max_depth:
                    successors = problem.get_successors(
                        node, self.num_successors_per_node
                    )
                    for successor in successors:
                        print(successor.action_skeleton_as_strings)
                    next_beam.extend(successors)
                else:
                    print(colored("Reached max depth, stopping search", "red"))
                    break

            if len(next_beam) == 0:
                print(colored("No successors, stopping search", "red"))
                break

            # Set the current beam to the next beam
            node_scores = problem.get_node_scores(next_beam)

            if visualize:
                potential_actions_as_strings: List[str] = [
                    str(node.action_primitive) for node in next_beam
                ]
                table_headers = ["Action", "LM", "Value", "Overall"]
                lm_action_scores = [node.action_sequence_score_lm for node in next_beam]
                value_action_scores = [
                    node.action_sequence_q_product_post_optimization
                    for node in next_beam
                ]
                formatted_table_str: str = format_saycan_scoring_table(
                    table_headers,
                    potential_actions_as_strings,
                    lm_action_scores,
                    value_action_scores,
                    node_scores,
                )
                custom_recording_text: str = f"""human: {problem.instruction}\npredicted: {problem.goal_props}\n{formatted_table_str}"""
                # set these values for each node as a record of what happened with the 'competitor' nodes
                for i in range(len(next_beam)):
                    next_beam[i].beam = next_beam
                    next_beam[i].custom_recording_text = custom_recording_text

                for node in next_beam:
                    planners.vizualize_predicted_plan(
                        node.action_skeleton_as_strings,
                        node.env,
                        node.action_skeleton_as_primitives,
                        node.motion_plan_post_optimization,
                        path=pathlib.Path("plots/ilm_tamp/"),
                        object_relationships_list=node.object_relationships_sequence_post_optimization,
                        custom_recording_text=node.custom_recording_text_sequence,
                        file_extensions=["mp4"],
                    )

            # check if any of the successors are a solution
            results: List[bool] = [problem.is_end(node) for node in next_beam]
            if any(results):
                return [
                    successor
                    for (successor, result) in zip(successors, results)
                    if result
                ]

            beam = self.filter_beam(
                next_beam, node_scores, max_beam_size=self.max_beam_size
            )

        return beam
