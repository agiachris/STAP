"""
Usage instructions:

python scripts/lm_token_calculation.py
"""

from enum import Enum

class ActionSequenceMethod(Enum):
    """Action sequence method."""
    LM_SHOOTING_VIA_REQUERYING = 0  # 'perspective 2': query the LM for entire action sequence but with different temperatures
    LM_SHOOTING_VIA_PROMPT_SPECIFICATION = 1  # same as above, except query model once and modify the prompt to ask LM to generation K action plans
    SCORING = 2  # 'saycan': assume SayCan's scoring + TAPS generates successfully action sequence with length 'search_depth'
    BEAM_SEARCH_SCORING = 3  # 'beam search': from the LM's POV, BEARM_SEARCH_SCORING is the same as SCORING, except now we have a beam width K instead of 1

def tokens_for_single_problem_goal_only(num_in_context: int):
    # compute cost of evaluating classifier (using only AND goal predicates)
    sample_header = """Available predicates: ['on(a, b)', 'inhand(a)']
    Available primitives: ['pick(a, b)', 'place(a, b)', 'pull(a, b)', 'push(a, b)']"""

    sample_input = """Available scene objects: ['table', 'rack', 'blue_box', 'red_box', 'yellow_box', 'cyan_box']
    Object relationships: ['on(rack, table)', 'on(blue_box, rack)', 'on(yellow_box, rack)', 'on(red_box, table)', 'on(cyan_box, yellow_box)']
    Human instruction: Hey there, put the blue_box on the top of the rack and place the yellow_box onto the rack and position the cyan_box on the table and move the red_box above the table now, thanks!
    Goal predicate list: """

    sample_output = """['on(blue_box, rack)', 'inhand(yellow_box)', 'on(red_box, table)', 'on(cyan_box, yellow_box)']"""

    avg_header_input = len(sample_header)
    avg_input_len = len(sample_input)
    avg_output_len = len(sample_output)

    entire_prompt_length = (
        avg_header_input
        + num_in_context * (avg_input_len + avg_output_len)
        + avg_input_len
    )
    avg_output_len = len(sample_output)
    return entire_prompt_length, avg_output_len


def tokens_for_single_problem_action_sequence(
    method: ActionSequenceMethod,
    num_in_context: int,
    search_depth: int = -1,
    num_skills: int = -1,
    num_lm_shooting_samples: int = -1,
    beam_width: int = -1,
):
    sample_header = """Available predicates: ['on(a, b)', 'inhand(a)']
    Available primitives: ['pick(a, b)', 'place(a, b)', 'pull(a, b)', 'push(a, b)']"""

    sample_input = """Available scene objects: ['table', 'rack', 'blue_box', 'red_box', 'yellow_box', 'cyan_box']
    Object relationships: ['on(rack, table)', 'on(blue_box, rack)', 'on(yellow_box, rack)', 'on(red_box, table)', 'on(cyan_box, yellow_box)']
    Human instruction: Hey there, put the blue_box on the top of the rack and place the yellow_box onto the rack and position the cyan_box on the table and move the red_box above the table now, thanks!
    Goal predicate list: ['on(blue_box, rack)', 'inhand(yellow_box)', 'on(red_box, table)', 'on(cyan_box, yellow_box)']
    Robot action sequence: """

    sample_full_sequence_output = """['pick(blue_box, rack)', 'place(yellow_box, rack)', 'place(cyan_box, table)', 'place(red_box, table)']"""

    avg_header_input = len(sample_header)
    avg_input_len = len(sample_input)
    avg_full_sequence_output_len = len(sample_full_sequence_output)
    avg_single_skill_len = len("pick(blue_box, rack)")

    entire_prompt_length = (
        avg_header_input
        + num_in_context * (avg_input_len + avg_full_sequence_output_len)
        + avg_input_len
    )

    if method.value == ActionSequenceMethod.LM_SHOOTING_VIA_REQUERYING.value:
        assert num_lm_shooting_samples > 0, "num_lm_shooting_samples must be > 0"
        input_tokens, output_tokens = (
            num_lm_shooting_samples * entire_prompt_length,
            num_lm_shooting_samples * avg_full_sequence_output_len,
        )
    elif (
        method.value == ActionSequenceMethod.LM_SHOOTING_VIA_PROMPT_SPECIFICATION.value
    ):
        assert num_lm_shooting_samples > 0, "num_lm_shooting_samples must be > 0"
        input_tokens, output_tokens = (
            entire_prompt_length,
            num_lm_shooting_samples * avg_full_sequence_output_len,
        )
    elif method.value == ActionSequenceMethod.SCORING.value:
        assert search_depth > 0, "search_depth must be > 0"
        n_calls = num_skills * search_depth
        input_tokens, output_tokens = (
            n_calls * entire_prompt_length,
            n_calls * avg_single_skill_len,
        )
    elif method.value == ActionSequenceMethod.BEAM_SEARCH_SCORING.value:
        assert beam_width > 0, "beam_width must be specified for beam search scoring"
        n_calls = num_skills * search_depth * beam_width
        input_tokens, output_tokens = (
            n_calls * entire_prompt_length,
            n_calls * avg_single_skill_len,
        )
    else:
        raise ValueError(f"Unknown method {method}")

    return input_tokens, output_tokens


def get_cost(tokens: int, engine: str = "davinci") -> float:
    if engine == "davinci":
        cost = tokens / 1000 * 0.02
    elif engine == "curie":
        cost = tokens / 1000 * 0.002
    elif engine == "babbage":
        cost = tokens / 1000 * 0.0005
    elif engine == "ada":
        cost = tokens / 1000 * 0.0004
    else:
        raise ValueError("engine must be one of davinci, curie, babbage, ada")
    return round(cost, 2)


def main():
    ENGINE = "davinci"
    NUM_IN_CONTEXT_EXAMPLES = 5
    BEAM_WIDTH = 3
    NUM_OBJECTS_PER_DOMAIN = 7
    NUM_LM_SHOOTING_SAMPLES = 5
    SEARCH_DEPTH = 6
    TOP_K_NEXT_ACTIONS = 3

    # assuming 4 predicates: pick (assume only need to specify the object) + place + pull + push
    num_skills = (
        NUM_OBJECTS_PER_DOMAIN
        + NUM_OBJECTS_PER_DOMAIN * 5  # place predicate need to specify location as well
        + NUM_OBJECTS_PER_DOMAIN
        + NUM_OBJECTS_PER_DOMAIN
    )

    if TOP_K_NEXT_ACTIONS > 0:
        num_skills = min(num_skills, TOP_K_NEXT_ACTIONS)

    NUM_DOMAINS = 3
    NUM_PROBLEMS_PER_DOMAIN = 3
    NUM_PROBLEMS_TOTAL = NUM_PROBLEMS_PER_DOMAIN * NUM_DOMAINS

    total_input_tokens = 0
    total_output_tokens = 0

    print("=== Single problem goal-only costs and tokens===")
    input_tokens, output_tokens = tokens_for_single_problem_goal_only(
        NUM_IN_CONTEXT_EXAMPLES
    )
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    total_tokens = input_tokens + output_tokens    
    print(
        f"Goal: total_tokens {total_tokens}, total_cost ${get_cost(total_tokens, ENGINE)}\n"
    )

    print("=== Single problem action sequence costs and tokens===")
    print("Assumptions:")
    print((
        f"objects per domain: {NUM_OBJECTS_PER_DOMAIN}\n"
        f"LM shooting samples: {NUM_LM_SHOOTING_SAMPLES}\n"
        f"search depth: {SEARCH_DEPTH}\n"
        f"beam width: {BEAM_WIDTH}\n"
        f"number of in context examples: {NUM_IN_CONTEXT_EXAMPLES}\n"
        f"engine: {ENGINE}\n"))

    for method in ActionSequenceMethod:
        input_tokens, output_tokens = tokens_for_single_problem_action_sequence(
            method,
            NUM_IN_CONTEXT_EXAMPLES,
            search_depth=SEARCH_DEPTH,
            num_skills=num_skills,
            num_lm_shooting_samples=NUM_LM_SHOOTING_SAMPLES,
            beam_width=BEAM_WIDTH,
        )
        total_tokens = input_tokens + output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        print(
            f"{method.name}: total_tokens {total_tokens}, total_cost ${get_cost(total_tokens, ENGINE)}"
        )

        if method.name == ActionSequenceMethod.LM_SHOOTING_VIA_PROMPT_SPECIFICATION.name:
            print(
                f"   - Manual prompting engineering shows some promise; need more testing]"
            )
        elif method.name == ActionSequenceMethod.SCORING.name:
            print(
                f"   - Assumes that beam-width = 1 (i.e. what SayCan does) works"
            )

    print(f"\nTotal input tokens for all methods: {total_input_tokens}")
    print(f"Total output tokens for all methods: {total_output_tokens}")

    # compute tokens for all problems and methods
    total_input_tokens *= NUM_PROBLEMS_TOTAL
    total_output_tokens *= NUM_PROBLEMS_TOTAL

    print(f"\nTotal input tokens for all problems and methods: {total_input_tokens}")
    print(f"Total output tokens for all problems and methods: {total_output_tokens}")

    # compute costs for all problems and all methods
    print(f"\nTotal cost for all problems and methods: ${get_cost(total_input_tokens + total_output_tokens, ENGINE)}")

if __name__ == "__main__":
    main()
