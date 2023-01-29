from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, Optional
import numpy as np

from termcolor import colored
from configs.base_config import LMConfig
from scripts.eval.eval_saycan import format_saycan_scoring_table
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.task_planners.lm_agent import LMPlannerAgent
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
    get_next_action_str_from_lm,
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
        all_ground_truth_prior_object_relationships: List[List[str]] = None,
        all_ground_truth_executed_actions: List[str] = None,
    ):
        """
        Args:
            parent: parent node
            action_primitive: action primitive that led to this node; set upon creation
            geometric_state: geometric state of the environment at this node
            motion_planner: motion planner used to generate this node
            env: environment
            available_predicates: available predicates
            all_ground_truth_prior_object_relationships: list of object relationships that have
                actually been seen (as opposed to the object relationships from the dynamics roll-out
                and then getting parsed) -- only used for the root node
            all_ground_truth_executed_actions: list of actions that have actually been executed (as
                opposed to the actions from the dynamics roll-out) --- only used for the root node

        For a given node, there are 2 types of environment observations:
            1. all ground truth environment observations
            2. all predicted environment observations (from dynamics roll-out from the root node)

        Information from the POV of the current node:

        gt = ground truth
        p = predicted

        gto1     gta1    gto2    gta2    gto3    pa1   po1    pa2     po2     pa3    po3
                                        ^                                   ^
                                        |                                   |
                                    root-node                           current-node
                                                ^
                                                |
                                            child-node-0
                                                                ^
                                                                |
                                                        child-node-1

        root node does not have action_primitive

        To generate actions for the current node, we need to know:
            1. current_object_relationships = po3 (if is root: gt03)
            2. all_executed_actions = [gta1, gta2, pa1, pa2, pa3] (if is root: [gta1, gta2])
            3. all_prior_object_relationships = [gto1, gto2, gto3, po1, po2, po3] (if is root: [gto1, gto2, gto3])
        """
        self.action_primitive = action_primitive
        self.parent = parent
        self._geometric_state = geometric_state
        self._motion_planner = motion_planner
        self._env = env
        self._available_predicates = available_predicates
        self._all_ground_truth_prior_object_relationships = (
            all_ground_truth_prior_object_relationships
            if all_ground_truth_prior_object_relationships is not None
            else []
        )
        self._all_ground_truth_executed_actions = (
            all_ground_truth_executed_actions
            if all_ground_truth_executed_actions is not None
            else []
        )
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

    @cached_property
    def root_node_object_relationships(self) -> List[str]:
        root_node_object_relationships = get_object_relationships(
            self.root_node_geometric_state,
            self.env.prop_testing_objs,
            self.available_predicates,
            use_hand_state=False,
        )
        return list(map(str, root_node_object_relationships))

    @cached_property
    def current_object_relationships(self) -> List[str]:
        if self.parent is None:
            root_node_object_relationships = get_object_relationships(
                self.root_node_geometric_state,
                self.env.prop_testing_objs,
                self.available_predicates,
                use_hand_state=False,
            )
            return list(map(str, root_node_object_relationships))
        return self.object_relationships_sequence_post_optimization[-1]

    @property
    def all_executed_actions(self) -> List[str]:
        if self.parent is None:
            return self._all_ground_truth_executed_actions
        return self.parent.all_executed_actions + self.action_skeleton_as_strings

    @property
    def all_ground_truth_prior_object_relationships(self) -> List[str]:
        if self.parent is None:
            return self._all_ground_truth_prior_object_relationships
        return self.parent.all_ground_truth_prior_object_relationships

    @property
    def all_prior_object_relationships(self) -> List[str]:
        if self.parent is None:
            return self.all_ground_truth_prior_object_relationships
        return (
            self.all_ground_truth_prior_object_relationships
            + self.object_relationships_sequence_post_optimization
        )

    @property
    def pre_action_current_object_relationships(self) -> List[str]:
        assert (
            self.parent is not None
        ), "pre_action_current_object_relationships is not defined for root node as root node has no action"
        return self.all_prior_object_relationships[-2]

    @property
    def pre_action_all_executed_actions(self) -> List[str]:
        return self.all_executed_actions[:-1]

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

    @property
    def pre_action_all_prior_object_relationships(self) -> List[str]:
        return self.all_prior_object_relationships[:-1]

    @action_sequence_score_lm.setter
    def action_sequence_score_lm(self, value: float):
        self._action_sequence_score_lm = value

    @property
    def action_sequence_q_product_post_optimization(self):
        return self.motion_plan_post_optimization.values.prod()

    @property
    def last_action_q_value_post_optimization(self):
        return self.motion_plan_post_optimization.values[-1]

    @property
    def root_node_geometric_state(self) -> np.ndarray:
        if self.parent is None:
            return self._geometric_state
        return self.parent.root_node_geometric_state

    @property
    def all_ground_truth_prior_object_relationships(self):
        """
        Returns:
            list of object relationships that have actually been seen (as opposed to object
            relationships parsed from the dynamics roll-out) -- only contained in the root node
        """
        if self.parent is None:
            return self._all_ground_truth_prior_object_relationships
        return self.parent.all_ground_truth_prior_object_relationships

    @property
    def all_ground_truth_executed_actions(self):
        """
        Returns:
            list of actions that have actually been executed (as opposed to actions from the
            dynamics roll-out) -- only contained in the root node
        """
        if self.parent is None:
            return self._all_ground_truth_executed_actions
        return self.parent.all_ground_truth_executed_actions

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
    def action_skeleton_as_strings(self) -> List[str]:
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

    @property
    def all_executed_actions(self) -> List[str]:
        """
        Returns:
            list of actions that have actually been executed as well as
            actions from the dynamics roll-out
        """
        if self.parent is None:
            return self._all_ground_truth_executed_actions
        return (
            self.parent.all_ground_truth_executed_actions
            + self.action_skeleton_as_strings
        )

    @cached_property
    def motion_plan_post_optimization(self) -> planners.PlanningResult:
        """Full motion plan after optimization"""
        return self.motion_planner.plan(
            self.root_node_geometric_state,
            instantiate_task_plan_primitives(self.action_skeleton_as_strings, self.env),
        )

    @property
    def object_relationships_sequence_post_optimization(self) -> List[List[str]]:
        """All object relationships in the sequence of actions after optimization.

        Does not include the object relationships in the initial state;
        initial state obj-rels are handled by the root node with its ground truth object relationships.

        If root node, returns empty list as there is no sequence of actions."""
        # for each state in the motion plan, get the object relationships
        object_relationships_sequence: List[List[str]] = []
        if self.parent is None:
            return object_relationships_sequence
        for state in self.motion_plan_post_optimization.states[1:]:
            object_relationships = get_object_relationships(
                state,
                self.env.prop_testing_objs,
                self.available_predicates,
                use_hand_state=False,
            )
            object_relationships_sequence.append(list(map(str, object_relationships)))
        return object_relationships_sequence

    @property
    def all_prior_object_relationships_sequence(self) -> List[List[str]]:
        """All object relationships in the sequence of actions after optimization.

        ***Also includes all object relationships seen since the start of execution***"""
        return (
            self.all_ground_truth_prior_object_relationships
            + self.object_relationships_sequence_post_optimization
        )

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
        prop_testing_objs: List[Object],
        goal_props_callable: Callable[[List[str]], bool],
        pddl_domain_file: str,
        pddl_problem_file: str,
        examples: List[Dict[str, Any]],
        lm_cfg: LMConfig,
        all_prior_object_relationships: List[List[str]],
        all_executed_actions: List[str],
        lm_agent: LMPlannerAgent = LMPlannerAgent,
        auth: Optional[Authentication] = None,
        lm_cache: Optional[Dict[str, str]] = None,
        lm_cache_file: Optional[str] = None,
        termination_method: Literal[
            "pred_instr_achieved", "goal_prop"
        ] = "pred_instr_achieved",
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
        self.env.prop_testing_objs = prop_testing_objs
        self.goal_props_callable = goal_props_callable
        self.pddl_domain_file = pddl_domain_file  # TODO(klin); this is used for testing task plans' feasibility
        self.pddl_problem_file = pddl_problem_file
        # these belong in some LM object
        self.examples = examples
        self.lm_cfg = lm_cfg
        self.auth = auth
        self.lm_cache = lm_agent.lm_cache
        self.lm_cache_file = lm_cache_file
        self.termination_method = termination_method
        self.lm_agent = lm_agent
        self.lm_cache = lm_agent.lm_cache

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
            all_ground_truth_prior_object_relationships=all_prior_object_relationships,
            all_ground_truth_executed_actions=all_executed_actions,
        )

    def is_end(self, node: Node) -> bool:
        """Returns True if the node is a goal state."""
        print(
            colored(
                f"is_end: {node.action_skeleton_as_strings}",
                "light_yellow",
            )
        )
        print(f"Action skeleton values: {node.motion_plan_post_optimization.values}")

        print(
            f"Object relationships: {node.object_relationships_sequence_post_optimization[-1]}"
        )
        if self.termination_method == "pred_instr_achieved":
            # prompt the LM for the next action to execute
            # if next action is stop, then we are done
            # otherwise, we are not done
            current_node_object_relationships = node.current_object_relationships
            current_node_all_executed_actions = node.all_executed_actions
            current_node_all_prior_object_relationships = (
                node.all_prior_object_relationships
            )
            next_action_str = self.lm_agent.get_next_action_str(
                current_node_all_prior_object_relationships,
                current_node_all_executed_actions,
                verbose=True,
                in_context_example_robot_format="python_list",
                robot_prompt="Instruction achieved (True/False): ",
            )
            lm_cache = self.lm_agent.lm_cache
            self.lm_cache = lm_cache
            if "True" in next_action_str:
                print(
                    colored(
                        f"Stop due to predicting True after: {node.action_skeleton_as_strings}",
                        "magenta",
                    )
                )
                print(
                    f"Object relationships: {node.object_relationships_sequence_post_optimization[-1]}"
                )
                return True
        elif self.termination_method == "goal_prop":
            if is_satisfy_goal_props(
                self.goal_props_callable,
                self.prop_testing_objs,
                node.motion_plan_post_optimization.states[-1],
                use_hand_state=False,
            ):
                print(
                    colored(
                        f"Stop due to satisfying goal props: {node.action_skeleton_as_strings}",
                        "magenta",
                    )
                )
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

        # TODO(klin): skip scoring some of the action sequences if we know the Q product is below some threshold
        potential_actions_str = [str(node.action_primitive).lower() for node in nodes]
        pre_action_current_object_relationships = nodes[
            0
        ].pre_action_current_object_relationships
        pre_action_all_executed_actions = nodes[0].pre_action_all_executed_actions
        pre_action_all_prior_object_relationships = nodes[
            0
        ].pre_action_all_prior_object_relationships
        print(
            f"pre_action_current_object_relationships: {pre_action_current_object_relationships}"
        )
        print(f"pre_action_all_executed_actions: {pre_action_all_executed_actions}")
        print(
            f"pre_action_all_prior_object_relationships: {pre_action_all_prior_object_relationships}"
        )
        lm_action_scores, lm_cache = get_action_scores_from_lm(
            self.env.instruction,
            potential_actions_str,
            self.goal_props,
            list(self.env.objects.keys()),
            pre_action_current_object_relationships,
            pre_action_all_prior_object_relationships,
            pre_action_all_executed_actions,
            self.pddl_domain_file,
            self.pddl_problem_file,
            examples=self.examples,
            lm_cfg=self.lm_cfg,
            auth=self.auth,
            lm_cache=self.lm_cache,
            lm_cache_file=self.lm_cache_file,
            custom_in_context_example_robot_format="python_list",
            custom_robot_action_sequence_format="python_list",
        )
        self.lm_cache = lm_cache
        self.lm_agent.lm_cache = lm_cache

        for i in range(len(nodes)):
            nodes[i].action_sequence_score_lm = lm_action_scores[i]

        return [best_values[i] * lm_action_scores[i] for i in range(len(nodes))]

    def get_successors(self, node: Node, num_successors: int = 5) -> List[Node]:
        """Returns num_successors successors of a given node."""
        new_nodes: List[Node] = []
        current_node_object_relationships = node.current_object_relationships
        current_node_all_executed_actions = node.all_executed_actions
        current_node_all_prior_object_relationships = (
            node.all_prior_object_relationships
        )
        print(f"Current node object relationships: {current_node_object_relationships}")
        print(f"Current node all executed actions: {current_node_all_executed_actions}")
        print(
            f"Current node all prior object relationships: {current_node_all_prior_object_relationships}"
        )
        self.lm_cfg.echo = False
        actions, lm_cache = get_next_actions_from_lm(
            self.instruction,
            self.goal_props,
            list(self.env.objects.keys()),
            current_node_object_relationships,
            current_node_all_prior_object_relationships,
            current_node_all_executed_actions,
            self.pddl_domain_file,
            self.pddl_problem_file,
            examples=self.examples,
            custom_in_context_example_robot_prompt="Top robot action sequence: ",
            custom_in_context_example_robot_format="python_list",
            custom_robot_prompt=f"Top {num_successors} next actions (python list using available scene objects; do not use newline): ",
            custom_robot_action_sequence_format="python_list",
            lm_cfg=self.lm_cfg,
            auth=self.auth,
            lm_cache=self.lm_cache,
            verbose=True,
        )
        self.lm_agent.lm_cache = lm_cache
        self.lm_cache = lm_cache
        self.lm_cfg.echo = True
        self.lm_cfg.engine = "code-davinci-002"

        # use try except to catch any errors in the action primitives
        potential_action_primitives: List[primitives.Primitive] = []
        for action in actions:
            try:
                potential_action_primitives.append(
                    self.env.get_primitive_info(action, self.env)
                )
            except Exception as e:
                print(f"Error in action primitive: {e}")
                pass

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
        visualize_path: str = None,
    ) -> List[Node]:
        """
        Solves a given problem using beam search.
        """
        beam: List[Node] = [problem.start_node]
        # problem.is_end(problem.start_node)
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
                    obj_rel_lst = [node.root_node_object_relationships]
                    obj_rel_lst.extend(
                        node.object_relationships_sequence_post_optimization
                    )
                    planners.vizualize_predicted_plan(
                        node.action_skeleton_as_strings,
                        node.env,
                        node.action_skeleton_as_primitives,
                        node.motion_plan_post_optimization,
                        path=visualize_path,
                        object_relationships_list=obj_rel_lst,
                        custom_recording_text=node.custom_recording_text_sequence,
                        file_extensions=["mp4"],
                    )

            # check if any of the successors are a solution
            results: List[bool] = [problem.is_end(node) for node in next_beam]

            # remove any successors/results with a latest node with action value less than 0.5
            filtered_results = []
            filtered_successors = []
            for (successor, result) in zip(next_beam, results):
                if successor.last_action_q_value_post_optimization > 0.45:
                    filtered_results.append(result)
                    filtered_successors.append(successor)

            if any(filtered_results):
                return [
                    successor
                    for (successor, result) in zip(
                        filtered_successors, filtered_results
                    )
                    if result
                ]

            beam = self.filter_beam(
                next_beam, node_scores, max_beam_size=self.max_beam_size
            )

        return beam
