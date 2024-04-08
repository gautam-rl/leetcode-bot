import asyncio
import re
from textwrap import dedent
from pydantic import BaseModel
from os import listdir
from multiprocessing.pool import Pool
import logging
from rich.logging import RichHandler

from pytest_invoker import PyTestInvoker
from chat import CodingAssistantChat

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class ParsedSolution(BaseModel):
    problem_description: str
    reference_solution: str


def load_solution_file(name: str) -> ParsedSolution:
    with open(f"leetcode/solutions/{name}.py", "r") as file:
        contents = file.read()
        # Split the contents of the file into the problem description and the reference solution
        problem_description, reference_solution = contents.split("#" * 48)[1:3]
        return ParsedSolution(
            problem_description=problem_description.strip(),
            reference_solution=reference_solution.strip(),
        )


def load_test_file(name: str):
    with open(f"leetcode/tests/test_{name}.py", "r") as file:
        contents = file.read()
        return contents


class TestEvalMetrics(BaseModel):
    test_name: str
    exit_code: int
    log_lines: str
    time_taken_to_execute_tests: float
    time_taken_by_ai: float
    proposed_solution: str
    tokens_used: int


def get_method_name(test_file_contents: str) -> str:
    # Grep test_file_contest solution.<method>, and extract the method.
    return re.search(r"solution\.(?P<method>\w+)", test_file_contents).group(0)  # type: ignore


async def solve_leetcode_problem(test_name: str) -> TestEvalMetrics:
    try:
        parsed_solution: ParsedSolution = load_solution_file(test_name)
        test_file = f"leetcode/tests/test_{test_name}.py"
        # Load the test file into a string.
        with open(test_file, "r") as file:
            test_file_contents = file.read()

        method_name = get_method_name(test_file_contents)

        # Plan:
        # 1. Attempt the naive GPT-3.5 solution.
        # 2. If that fails, use a separate prompt to explain the failure + ask for a corrected solution.
        # 3. Loop over (2) with GPT-4 until we hit max attempts.

        candidate_solution = ""
        attempt = 0
        chat = CodingAssistantChat()
        pytest_invoker = PyTestInvoker(test_name, test_file_contents)

        while attempt <= 2:
            log.info("===========================================================")
            log.info(f"Attempt {attempt}")
            # Try a naive initial solution.
            if attempt == 0:
                provide_utils_context = ""
                if "ListNode" in test_file_contents:
                    # TODO - generate these automatically
                    provide_utils_context += dedent(
                        """\
                        You have access to the utils.list.ListNode class with this definition:

                        ```python
                        class ListNode:
                            def __init__(self, x):
                                self.val = x
                                self.next = None
                        ```

                        You can access it using `import utils.list.ListNode`
                        Do not declare it in your response.
                        """
                    )
                if "TreeNode" in test_file_contents:
                    provide_utils_context += dedent(
                        """\
                        You have access to the utils.tree.TreeNode class with this definition:

                        ```python
                        class TreeNode:
                            def __init__(self, x):
                                self.val = x
                                self.left = None
                                self.right = None
                        ```

                        You can access it using `import utils.tree.TreeNode`
                        Do not declare it in your response.
                        """
                    )
                completion: str = await chat.generate_completion(
                    f"""
                    Please solve the following problem in standard python without third-party libraries.
                    The response should only contain valid python3.12 code including all imports and types.
                    Only respond with the code solution. Your response will be evaluated directly by the python code interpreter.
                    {provide_utils_context}

                    The solution should be of the form:
                    ```python
                    # Import things here
                    from typing import List
                    class Solution:
                        def {method_name}(self, <args go here>) -> <return type>:
                            # Solution goes here
                    ```
                    Problem:
                    {parsed_solution.problem_description}

                    Your Solution:
                    """
                )
            else:
                # The previous attempt failed. We need to provide more context to the AI.
                completion: str = await chat.generate_completion(
                    f"""
                    The provided solution failed the test cases. Please explain what went wrong and propose a new solution.
                    
                    pytest failure: {pytest_invoker.log_lines()}
                    """
                )

            # The response code is often wrapped in markdown and some explanation. Extract just the inner code.
            candidate_solution = completion
            candidate_solution = re.sub(r".*```python", "", candidate_solution, flags=re.S)
            candidate_solution = re.sub(r"```.*", "", candidate_solution, flags=re.S)
            # candidate_solution = completion.replace("```python", "").replace("```", "").strip()
            log.debug(f"Attempt {attempt}: {candidate_solution}")

            # Run the test.
            await pytest_invoker.run_test(candidate_solution)
            if pytest_invoker.exit_code() == 0:
                break
            attempt += 1

        return TestEvalMetrics(
            proposed_solution=candidate_solution,
            test_name=test_name,
            exit_code=pytest_invoker.exit_code(),
            log_lines=pytest_invoker.log_lines(),
            time_taken_to_execute_tests=pytest_invoker.elapsed_time(),
            time_taken_by_ai=chat.ai_time_elapsed(),
            tokens_used=chat.tokens_used(),
        )
    except Exception as e:
        log.error(f"Test {test_name} failed with error: {e}")
        return TestEvalMetrics(
            proposed_solution="",
            test_name=test_name,
            exit_code=1,
            log_lines=str(e),
            time_taken_to_execute_tests=0,
            time_taken_by_ai=0,
            tokens_used=0,
        )


class TotalEvalMetrics(BaseModel):
    test_eval_metrics: list[TestEvalMetrics]
    total_passed: int
    total_failed: int


def is_valid_test_file(test_file: str) -> bool:
    if test_file == "test_a0000blank.py":
        return False
    with open("leetcode/tests/" + test_file, "r") as f:
        contents = f.read()
    if "solution." not in contents:
        # TODO - need to write custom classes
        log.debug(f"Skipping {test_file} as it does not contain a Solution class")
        return False
    if "import Solution, Node" in contents:
        # TODO - handle this case
        log.debug(f"Skipping {test_file} as it uses custom classes.")
        return False
    return True


async def main():
    # Walk the test directory. For each test, generate a solution and verify with a test.
    test_names = []
    for testlfile in listdir("leetcode/tests"):
        if (
            testlfile.startswith("test_")
            and testlfile.endswith(".py")
            and is_valid_test_file(testlfile)
        ):
            test_names.append(testlfile[5:-3])

    # Only run a subset
    # test_names = sorted(test_names)[1:10]
    # test_names = ["a0001twosum"]
    # test_names = ["a0063uniquepathsii"]
    # test_names = ["a0092reverselinkedlistii"]

    log.info(f"Running {len(test_names)} tests")
    async_results = []
    results: list[TestEvalMetrics] = []
    # NB: ThreadPool does not play nicely with pytest.main()
    #pool = Pool(processes=64)
    tasks = []
    for test_name in test_names:
        # Create an async task for each
        tasks += [(test_name, asyncio.create_task(solve_leetcode_problem(test_name)))]
        #results += [asyncio.run(solve_leetcode_problem(test_name))]
        # Spawn a thread
        #async_result = pool.apply_async(solve_leetcode_problem, args=(test_name,))
        #async_results += [(test_name, async_result)]
        # Fake async, to debug multiprocessing issues.
        # async_results += [(test_name, solve_leetcode_problem(test_name))]

    # Wait for all threads
    log.info("Waiting for tests to complete...")
    for test_name, task in tasks:
        try:
            # Await all results
            results += [await task]
            #results += [async_result.get()]
            # Fake async
            # results += [async_result]
        except Exception as e:
            log.error(f"Test {test_name} failed with error: {e}")
            results += [
                TestEvalMetrics(
                    test_name=test_name,
                    exit_code=1,
                    log_lines=str(e),
                    time_taken_to_execute_tests=0,
                    time_taken_by_ai=0,
                    proposed_solution="",
                    tokens_used=0,
                )
            ]

    total_passed = 0
    total_failed = 0
    for result in results:
        if result.exit_code == 0:
            total_passed += 1
        else:
            total_failed += 1
    total_eval_metrics = TotalEvalMetrics(
        test_eval_metrics=results, total_passed=total_passed, total_failed=total_failed
    )
    for result in results:
        if result.exit_code != 0:
            # Convert to log multiline
            log.error(
                dedent(f"""\
                       ============= Test {result.test_name} Failed =============
                       {result.log_lines}

                       Proposed Solution:
                       {result.proposed_solution}
                       ===========================================================
                       """)
            )
    log.info(
        dedent(f"""\
          Summary:
            Total Passed: {total_passed}
            Total Failed: {total_failed}
            Elapsed AI time: {sum([r.time_taken_by_ai for r in results]):.2f}s
            Elapsed pytest time: {sum([r.time_taken_to_execute_tests for r in results]):.2f}s
            AI tokens used: {sum([r.tokens_used for r in results])} (${sum([r.tokens_used for r in results])  * 0.5 / 1e6:.4f})
          """)
    )

if __name__ == "__main__":
    asyncio.run(main())