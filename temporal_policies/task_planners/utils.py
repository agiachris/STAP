import heapq
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

from temporal_policies.envs.base import Env, Primitive

########################################################################################
# Abstract Interfaces for State, Search Problems, and Search Algorithms.


@dataclass(frozen=True, order=True)
class State:
    """
    Usually, the state would be the current observation of the environment,
    since that would usually be enough to determine the next action to take.

    However, in this case, we need to keep track of
    (a) the action sequence that led to this state
    (b) the action distribution for the action sequence that leads
        to maximum Q product up to the current action primitive
    (c) the CEM planner, the env,

    so that we can:

    i) reconstruct the optimal action sequence at the end of the search
        (a)(b)
    ii) perform TAPS optimization on the action sequence
        (b)
    iii) use the LM to score the action sequence as a whole
        (a)(c: env to construct the observation as a string)

    In our case, the state is really just the sequence of action primitives taken
    up til this particular 'state'. Usually, the state is
    just the actual environment state (i.e. the poses of
    every object) in the scene (rolled forward using a "dynamics" model)

    In this case, the search problem would contain the info
    required to get dynamics.  As you implement different types of search problems throughout the assignment,
    think of what `memory` should contain to enable efficient search!
    """

    _action_primitve: Primitive
    optimized_action_sequence_observation: str
    _env: Env
    parent: "State"  # awkward to define the state as having a parent
    # since State implies everything required is inside the state?
    #  maybe it's fair to define the state recurisvely


class SearchProblem:
    # Return the start state.
    def startState(self) -> State:
        raise NotImplementedError("Override me")

    # Return whether `state` is an end state or not.
    def isEnd(self, state: State) -> bool:
        raise NotImplementedError("Override me")

    # Return a list of (action: str, state: State, cost: float) tuples corresponding to
    # the various edges coming out of `state`
    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        raise NotImplementedError("Override me")


class SearchAlgorithm:
    def __init__(self):
        """
        A SearchAlgorithm is defined by the function `solve(problem: SearchProblem)`

        A call to `solve` sets the following instance variables:
            - self.actions: List of "actions" that takes one from the start state to a
                            valid end state, or None if no such action sequence exists.
                            > Note: For this assignment, an "action" is just the string
                                    "nextLocation" for a state, but in general, an
                                    action could be something like "up/down/left/right"

            - self.pathCost: Sum of the costs along the path, or None if no valid path.

            - self.numStatesExplored: Number of States explored by the given search
                                      algorithm as it attempts to find a satisfying
                                      path. You can use this to gauge the efficiency of
                                      search heuristics, for example.

            - self.pastCosts: Dictionary mapping each State location visited by the
                              SearchAlgorithm to the corresponding cost to get there
                              from the starting location.
        """
        self.actions: List[str] = None
        self.pathCost: float = None
        self.numStatesExplored: int = 0
        self.pastCosts: Dict[State, float] = {}

    def solve(self, problem: SearchProblem) -> None:
        raise NotImplementedError("Override me")


class Heuristic:
    # A Heuristic object is defined by a single function `evaluate(state)` that
    # returns an estimate of the cost of going from the specified `state` to an
    # end state. Used by A*.
    def evaluate(self, state: State) -> float:
        raise NotImplementedError("Override me")


########################################################################################
# Uniform Cost Search (Dijkstra's algorithm)


class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose: int = 0):
        super().__init__()
        self.verbose = verbose

    def solve(self, problem: SearchProblem) -> None:
        """
        Run Uniform Cost Search on the specified `problem` instance.

        Sets the following instance variables (see `SearchAlgorithm` docstring).
            - self.actions: List[str]
            - self.pathCost: float
            - self.numStatesExplored: int
            - self.pastCosts: Dict[str, float]

        *Hint*: Some of these might be really helpful for Problem 3!
        """
        self.actions: List[str] = None
        self.pathCost: float = None
        self.numStatesExplored: int = 0
        self.pastCosts: Dict[State, float] = {}

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # Map state -> previous state.

        # Add the start state
        startState = problem.startState()
        frontier.update(startState, 0.0)

        while True:
            # Remove the state from the queue with the lowest pastCost (priority).
            state, pastCost = frontier.removeMin()
            if state is None and pastCost is None:
                if self.verbose >= 1:
                    print("Searched the entire search space!")
                return

            # Update tracking variables
            self.pastCosts[state.location] = pastCost
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print(f"Exploring {state} with pastCost {pastCost}")

            # Check if we've reached an end state; if so, extract solution.
            if problem.isEnd(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.pathCost = pastCost
                if self.verbose >= 1:
                    print(f"numStatesExplored = {self.numStatesExplored}")
                    print(f"pathCost = {self.pathCost}")
                    print(f"actions = {self.actions}")
                return

            # Expand from `state`, updating the frontier with each `newState`
            for action, newState, cost in problem.successorsAndCosts(state):
                if self.verbose >= 3:
                    print(f"\t{state} => {newState} (Cost: {pastCost} + {cost})")

                if frontier.update(newState, pastCost + cost):
                    # We found better way to go to `newState` --> update backpointer!
                    backpointers[newState] = (action, state)


# Data structure for supporting uniform cost search.
class PriorityQueue:
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert `state` into the heap with priority `newPriority` if `state` isn't in
    # the heap or `newPriority` is smaller than the existing priority.
    #   > Return whether the priority queue was updated.
    def update(self, state: State, newPriority: float) -> bool:
        oldPriority = self.priorities.get(state)
        if oldPriority is None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority) or (None, None) if empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                # Outdated priority, skip
                continue
            self.priorities[state] = self.DONE
            return state, priority

        # Nothing left...
        return None, None
