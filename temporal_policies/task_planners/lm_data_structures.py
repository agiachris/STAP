import ast
from dataclasses import asdict, dataclass
import pathlib
from typing import List, Literal, Optional
import openai
import json

# complication: Episode can be used as a single CoT prompt, or as a prompt for current context
# complication: Human might give an impossible goal ...
# Single prompt unit that can be used to compose a chain of thought prompt or a current context prompt

SCENE_OBJECT_PROMPT = "Available scene objects: "
SCENE_OBJECT_RELATIONSHIP_PROMPT = "Object relationships: "
SCENE_PREDICATE_PROMPT = "Available predicates: "
SCENE_PRIMITIVE_PROMPT = "Available primitives: "
HUMAN_INSTRUCTION_PROMPT = "Human instruction: "
EXPLANATION_PROMPT = "Explanation: "
GOAL_PROMPT = (
    "Goal predicate set: "  # set seems to work better than list? nevermind ... hmm
)
ROBOT_PROMPT = "Robot action sequence: "


def register_openai_key(key: str) -> None:
    openai.api_key = key


# remove because can't save Enum in json
# class RobotAnswerFormat(Enum):
#     PYTHON_LIST = 0
#     PYTHON_LIST_OF_LISTS = 1


# Note(klin): redundant with InContextExampleConfig because
# otherwise can't save to json
@dataclass
class InContextExample:
    predicates: Optional[List] = None
    primitives: Optional[List] = None
    scene_objects: Optional[List[str]] = None
    scene_object_relationships: Optional[List[str]] = None
    human: Optional[str] = None
    explanation: Optional[str] = None
    goal: Optional[List[str]] = None
    robot: Optional[List[str]] = None

    use_primitives: bool = False
    use_predicates: bool = False
    use_scene_objects: bool = False
    use_scene_object_relationships: bool = False
    use_human: bool = False
    use_explanation: bool = False
    use_goal: bool = False
    use_robot: bool = False

    # storage for predicted values
    explanation_predicted: Optional[str] = None  # explanation predicted from GPT3
    goal_predicted: Optional[str] = None  # goal predicted can be generated via scoring
    action_predicted: Optional[
        str
    ] = None  # action predicted can be generated via scoring
    robot_predicted: Optional[str] = None

    # original PDDL domain and problem files
    pddl_domain_file: Optional[str] = None
    pddl_problem_file: Optional[str] = None

    # custom prompt engineering configs
    custom_robot_prompt: str = ""
    custom_robot_answer_format: Literal[
        "python_list", "python_list_of_lists"
    ] = "python_list"  # use special prompt for robot action sequence in overall_example
    
    @property
    def action_prompt(self) -> str:
        """
        Format the prompt to get actions.

        primitive: <list of primitives>
        predicate: <list of predicates>
        available scene objects: <list of available scene objects>
        object relationships: <list of object relationships>
        human: <human utterance>

        Returns: [str] <prompt for 'explanation'>
        """
        res = ""
        if self.use_primitives:
            res += f"{SCENE_PRIMITIVE_PROMPT}{self.primitives}\n"
        if self.use_predicates:
            res += f"{SCENE_PREDICATE_PROMPT}{self.predicates}\n"

        return res + (
            f"{SCENE_OBJECT_PROMPT}{self.scene_objects}\n"
            f"{SCENE_OBJECT_RELATIONSHIP_PROMPT}{self.scene_object_relationships}\n"
            f"{HUMAN_INSTRUCTION_PROMPT}{self.human}\n"
            f"{EXPLANATION_PROMPT}{self.explanation}\n"
            f"{GOAL_PROMPT}{str(self.goal)}\n"
            f"{ROBOT_PROMPT}"
        )

    # technically, goal prompt can be generated via scoring
    @property
    def goal_prompt(self) -> str:
        """
        Format the prompt to get a goal.

        primitive: <list of primitives>
        predicate: <list of predicates>
        available scene objects: <list of available scene objects>
        object relationships: <list of object relationships>
        human: <human utterance>

        Returns: [str] <prompt for 'explanation'>
        """
        res = ""
        if self.use_primitives:
            res += f"{SCENE_PRIMITIVE_PROMPT}{self.primitives}\n"
        if self.use_predicates:
            res += f"{SCENE_PREDICATE_PROMPT}{self.predicates}\n"

        return res + (
            f"{SCENE_OBJECT_PROMPT}{self.scene_objects}\n"
            f"{SCENE_OBJECT_RELATIONSHIP_PROMPT}{self.scene_object_relationships}\n"
            f"{HUMAN_INSTRUCTION_PROMPT}{self.human}\n"
            f"{EXPLANATION_PROMPT}{self.explanation}\n"
            if self.explanation
            else "" f"{GOAL_PROMPT}"
        )

    @property
    def explanation_prompt(self) -> str:
        """
        Format the prompt to get an explanation.

        primitive: <list of primitives>
        predicate: <list of predicates>
        available scene objects: <list of available scene objects>
        object relationships: <list of object relationships>
        human: <human utterance>

        Returns: [str] <prompt for 'explanation'>
        """
        res = ""
        if self.use_primitives:
            res += f"{SCENE_PRIMITIVE_PROMPT}{self.primitives}\n"
        if self.use_predicates:
            res += f"{SCENE_PREDICATE_PROMPT}{self.predicates}\n"

        return res + (
            f"{SCENE_OBJECT_PROMPT}{self.scene_objects}\n"
            f"{SCENE_OBJECT_RELATIONSHIP_PROMPT}{self.scene_object_relationships}\n"
            if self.scene.get("relationships")
            else "" f"{HUMAN_INSTRUCTION_PROMPT}{self.human}\n" f"{EXPLANATION_PROMPT}"
        )

    @property
    def overall_example(self) -> str:
        """
        Format the entire example to be used as part of a chain of thought prompt.

        primitive: <list of primitives>
        predicate: <list of predicates>
        available scene objects: <list of available scene objects>
        object relationships: <list of object relationships>
        human: <human utterance>
        explanation: <explanation>
        goal predicate list: <goal list>
        robot: <list of robot>

        Returns: [str] <single prompt>
        """
        res = ""
        if self.use_primitives and self.primitives is not None:
            res += f"{SCENE_PRIMITIVE_PROMPT}{self.primitives}\n"
        if self.use_predicates and self.predicates is not None:
            res += f"{SCENE_PREDICATE_PROMPT}{self.predicates}\n"
        if self.use_scene_objects and self.scene_objects is not None:
            res += f"{SCENE_OBJECT_PROMPT}{self.scene_objects}\n"
        if (
            self.use_scene_object_relationships
            and self.scene_object_relationships is not None
        ):
            res += (
                f"{SCENE_OBJECT_RELATIONSHIP_PROMPT}{self.scene_object_relationships}\n"
            )
        if self.use_human and self.human is not None:
            res += f"{HUMAN_INSTRUCTION_PROMPT}{self.human}\n"
        if self.use_explanation and self.explanation is not None:
            res += f"{EXPLANATION_PROMPT}{self.explanation}\n"
        if self.use_goal and self.goal is not None:
            res += f"{GOAL_PROMPT}{str(self.goal)}\n"
        if self.use_robot and self.robot is not None:
            robot_prompt = (
                self.custom_robot_prompt
                if self.custom_robot_prompt != ""
                else ROBOT_PROMPT
            )
            answer = (
                f"[{self.robot}, ]"
                if self.custom_robot_answer_format == "python_list_of_lists"
                else self.robot
            )
            res += f"{robot_prompt}{answer}\n"
        return res

    def save_to_json(self, path: str, overwrite: bool = False) -> None:
        print(f"Saving to {path} ...")
        if not pathlib.Path(path).exists() or overwrite:
            with open(path, "w") as f:
                json.dump([asdict(self)], f)
        else:
            with open(path, "r") as f:
                # load existing data
                data = json.load(f)
                if type(data) == list:
                    # append new data
                    data.append(asdict(self))
                else:
                    raise ValueError(f"File {path} is not a list of results.")
            with open(path, "w") as f:
                json.dump(data, f)


@dataclass
class CurrentExample(InContextExample):
    predict_human: bool = False  # whether to predict human utterance :)
    predict_goal: bool = False
    predict_robot: bool = False
    predict_explanation: bool = False
    use_predicted_goal: bool = False

    def create_from_incontext_example(self, inc_example: InContextExample):
        # create a CurrentExample from an InContextExample
        for k, v in asdict(inc_example).items():
            setattr(self, k, v)

    # def __post_init__(self):
    #     if self.use_predicted_goal:
    #         assert (
    #             self.predict_goal
    #         ), "Cannot use predicted goal if not predicting goal."


@dataclass
class Result:
    header_prompt: Optional[InContextExample] = None
    examples: Optional[List[InContextExample]] = None
    scene_objects: Optional[str] = None
    scene_object_relationships: Optional[str] = None
    predicates: Optional[str] = None
    primitives: Optional[str] = None
    human: Optional[str] = None
    explanation_predicted: Optional[str] = None
    goal_ground_truth: Optional[str] = None
    goal_predicted: Optional[str] = None
    robot_ground_truth: Optional[str] = None
    robot_predicted: Optional[str] = None
    engine: str = "n/a"

    use_predicted_goal: bool = False  # whether to use predicted goal in the prompt or to use the 'ground truth' goal

    test_goal: bool = False  # whether to compare LM's goal with the "ground truth"
    goal_success: bool = False
    test_robot: bool = (
        False  # whether to compare LM's robot's action sequence with the "ground truth"
    )
    robot_success: bool = False
    robot_prediction_result_types: Optional[List[Literal[
        "success: partial",
        "success",
        "failure: invalid symbolic action",
        "failure: misses goal",
    ]]] = None
    predicted_task_plan_descriptions: Optional[List[str]] = None
    custom_robot_prompt: Optional[str] = None
    custom_robot_answer_format: Optional[str] = None

    @property
    def parsed_goal_predicted(self) -> List[str]:
        """
        Parse the goal predicted from GPT3.

        Note: this parsing method assumes the (chain of thought)
        prompt's formatting encourages GPT3 to output a string
        representation of a list of strings. That is,
        this method will fail if GPT3 outputs a different format.
        The method that's more likely to generate admissible outputs
        is the scoring-of-admissible-options method.

        Alternatively, we could continue with the current assumptions and some
        cosine-similarity metric to retrieve the most similar admissible option.
        If we use this option, we may need to format the examples so the
        goals are natural text (e.g. "the red block is on the blue block"), rather
        than code text (e.g. "on(red block, blue block)") because, embeddings of natural text
        from generic models have better behaved cosine similarities.

        Returns: [List[str]] <parsed goal predicted>
        """
        assert self.goal_predicted is not None, "Goal predicted is None."
        return ast.literal_eval(self.goal_predicted.strip())

    @property
    def parsed_robot_predicted(self) -> List[str]:
        """
        Parse the robot predicted from GPT3.

        The note in parsed_goal_predicted applies here as well.

        Returns: [List[str]] <parsed robot predicted>
        """
        assert self.robot_predicted is not None, "Robot predicted is None."
        return ast.literal_eval(self.robot_predicted.strip())

    @property
    def parsed_robot_predicted_list_of_lists(self) -> List[str]:
        """
        Parse the robot predicted from GPT3.

        The note in parsed_goal_predicted applies here as well.

        Returns: [List[str]] <parsed robot predicted>
        """
        assert self.robot_predicted is not None, "Robot predicted is None."
        parsed = ast.literal_eval(self.robot_predicted.strip())
        assert type(parsed) == list, "Robot predicted is not a list."
        assert all(
            type(x) == list for x in parsed
        ), "Robot predicted is not a list of lists."
        return parsed

    def save_to_json(self, path: str, overwrite: bool = False) -> None:
        if not pathlib.Path(path).exists() or overwrite:
            with open(path, "w") as f:
                json.dump([asdict(self)], f)
        else:
            with open(path, "r") as f:
                # load existing data
                data = json.load(f)
                if type(data) == list:
                    # append new data
                    data.append(asdict(self))
                else:
                    raise ValueError(f"File {path} is not a list of results.")
            with open(path, "w") as f:
                json.dump(data, f)
