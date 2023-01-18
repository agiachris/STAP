import os
import argparse
import symbolic


def is_problem_file(filename: str, problem_postfix: str):
    if "task" in filename and problem_postfix in filename:
        return True
    return False


def main(
    domain_file: str,
    task_id: int = -1,
    max_plans: int = 1,
    max_depth: int = 10,
    timeout: float = 10.0,
    verbose: bool = False,
):
    root_dir = os.path.dirname(domain_file)
    problem_postfix = os.path.split(domain_file)[-1].split("_")[0]
    filenames = os.listdir(root_dir)

    problem_filenames = sorted(
        [
            os.path.join(root_dir, f)
            for f in filenames
            if is_problem_file(f, problem_postfix)
        ]
    )
    if task_id >= 0:
        problem_filenames = [problem_filenames[task_id]]

    for problem_file in problem_filenames:
        print(f"---\nDomain file: {domain_file}")
        print(f"Problem file: {problem_file}")

        pddl = symbolic.Pddl(domain_file, problem_file)
        planner = symbolic.Planner(pddl, pddl.initial_state)
        bfs = symbolic.BreadthFirstSearch(
            planner.root, max_depth=max_depth, timeout=timeout, verbose=verbose
        )

        plan_count = 0
        for plan in bfs:
            if plan_count >= max_plans:
                break
            plan_count += 1
            actions = [node.action for node in plan[1:]]
            print(f"\nPlan {plan_count}: {actions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain-file", type=str, required=True, help="Path to PDDL domain file."
    )
    parser.add_argument(
        "--task-id", type=int, default=-1, help="Task ID number, 0-N, or -1 for all."
    )
    parser.add_argument(
        "--max-plans", type=int, default=1, help="Maximum number of solutions to print."
    )
    parser.add_argument(
        "--max-depth", type=int, default=10, help="Maximum search depth."
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Planning timeout.")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose planning.")
    args = parser.parse_args()
    main(**vars(args))
