import os
import re
from textwrap import dedent
import uuid
from pydantic import BaseModel
import pytest
import openai
from io import StringIO
from contextlib import redirect_stdout
import time
from os import listdir
from multiprocessing.pool import Pool
import logging
from pprint import pformat
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

client = openai.Client()

BASE_TESTDIR = "/tmp/leetcode-bot"


class CodingAssistantChat(BaseModel):
    """
    Represents the full chat history. This lets us provide previous context to the AI.
    """

    _messages: list[dict[str, str]] = []
    # TODO: track gpt-3.5 vs gpt-4
    _tokens_used: int = 0
    _time_elapsed: float = 0

    def __init__(self):
        """
        Initialize the chat history with a system message.
        """
        super().__init__()
        self._messages = [
            {
                "role": "system",
                "content": "You are an expert python coder that specializes in writing clean code.",
            },
        ]

    def generate_completion(self, content: str, model="gpt-3.5-turbo") -> str:
        # Add the user message to the chat history.
        self._add_user_message(content)

        log.debug("Sending chat:")
        log.debug(pformat(self._messages))

        begin_time = time.time()
        # TODO - Summarize older messages if the context gets too large.
        assistant_response = client.chat.completions.create(
            messages=self._messages,  # type: ignore
            model=model,
            temperature=0,
        )

        log.debug(pformat(assistant_response.choices[0].message.content))

        # Record the time.
        self._time_elapsed += time.time() - begin_time

        choice0 = assistant_response.choices[0]
        if choice0.message.content:
            self._add_assistant_message(choice0.message.content)
        if assistant_response.usage:
            self._tokens_used += assistant_response.usage.total_tokens
        return choice0.message.content if choice0.message.content else ""

    def tokens_used(self) -> int:
        return self._tokens_used

    def ai_time_elapsed(self) -> float:
        return self._time_elapsed

    def _add_user_message(self, content: str):
        """
        Add a user message to the chat history.
        """
        self._messages.append({"role": "user", "content": content})

    def _add_assistant_message(self, content: str):
        """
        Add an assistant message to the chat history.
        """
        self._messages.append({"role": "assistant", "content": content})


class PyTestInvoker(BaseModel):
    """
    Invokes pytest on the given solution and test file.
    Captures all results.
    """

    _elapsed_time: float = 0
    _exit_code: int = -1
    _log_lines: str = ""

    def __init__(self, test_name: str, test_file_contents: str):
        super().__init__()
        self._test_name = test_name
        self._test_file_contents = test_file_contents

    def run_test(self, candidate_solution):
        # Replace the import ... with our Solution class from candidate_solution
        test_file_with_candidate = self._test_file_contents.replace(
            f"from solutions.{self._test_name} import Solution", candidate_solution
        )

        # Record the start time
        start_time = time.time()

        # Write the file to /tmp and run pytest.
        TESTDIR = f"{BASE_TESTDIR}/{self._test_name}"
        # NOTE: We have to randomize the filename because pytest internally caches the file :/
        tmp_testfile = f"{TESTDIR}/test_{self._test_name}_{uuid.uuid4()}.py"
        os.system(f"rm -rf {TESTDIR}")
        os.system(f"mkdir -p {TESTDIR}")
        with open(tmp_testfile, "w") as file:
            file.write(test_file_with_candidate)

        # Run pytest and capture the exit code + result to a str.
        log.info(f"Running test {self._test_name}...")
        temp_stdout = StringIO()
        with redirect_stdout(temp_stdout):
            self._exit_code = pytest.main(["-v", tmp_testfile, f"--rootdir={TESTDIR}"])
            self._log_lines = temp_stdout.getvalue()

        # Run using os.system
        # result: subprocess.CompletedProcess = subprocess.run(
        #   ["pytest", "-v", tmp_testfile, f"--rootdir={TESTDIR}"], capture_output=True
        # )
        # self._exit_code = result.returncode
        # self._log_lines = result.stdout.decode("utf-8")

        # DEBUG: Uncomment to fake passing
        # exit_code = 0
        # log_lines: str = "PASSED"

        if self._exit_code != 0:
            log.info(f"Test {self._test_name} failed with exit code {self._exit_code}")
        else:
            log.info(f"Test {self._test_name} passed")
        self._elapsed_time += time.time() - start_time

    def exit_code(self) -> int:
        """
        Return the exit code from the most recent run.
        0 means success, non-zero means failure. See https://docs.pytest.org/en/stable/reference/exit-codes.html
        """
        return self._exit_code

    def log_lines(self) -> str:
        """
        Verbose log output from the most recent run.
        """
        # TODO - strip out headers/junk
        return self._log_lines

    def elapsed_time(self) -> float:
        """
        Total elapsed time across all runs.
        """
        return self._elapsed_time


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


def solve_leetcode_problem(test_name: str) -> TestEvalMetrics:
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
                completion = chat.generate_completion(
                    f"""
                    Please solve the following problem in standard python without third-party libraries.
                    The response should only contain valid python3.12 code including all imports and types.
                    Only respond with the code solution. Your response will be evaluated directly by the python code interpreter.
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
                completion = chat.generate_completion(
                    f"""
                    The provided solution failed the test cases. Please explain what went wrong and propose a new solution.
                    
                    pytest failure: {pytest_invoker.log_lines()}
                    """
                )

            # The response code is often wrapped in markdown and some explanation. Extract just the inner code.
            candidate_solution = completion
            candidate_solution = re.sub(
                r".*```python", "", candidate_solution, flags=re.S
            )
            candidate_solution = re.sub(r"```.*", "", candidate_solution, flags=re.S)
            # candidate_solution = completion.replace("```python", "").replace("```", "").strip()
            log.debug(f"Attempt {attempt}: {candidate_solution}")

            # Run the test.
            pytest_invoker.run_test(candidate_solution)
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
        return False
    # We don't want to run tests that depend on custom imports.
    if "from utils.list.ListNode import ListNode" in contents:
        return False
    if "from utils.tree.TreeNode import TreeNode" in contents:
        return False
    if "from utils.listutil import ListUtil" in contents:
        return False
    if re.search(r"import.*Node", contents):
        return False
    return True


if __name__ == "__main__":
    # Clear and re-create the output directory.
    os.system(f"rm -rf {BASE_TESTDIR}")
    os.system(f"mkdir -p {BASE_TESTDIR}")

    # Iterate the test directory
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

    log.info(f"Running {len(test_names)} tests")
    async_results = []
    results: list[TestEvalMetrics] = []
    # NB: ThreadPool does not play nicely with pytest.main()
    pool = Pool(processes=64)
    for test_name in test_names:
        # Spawn a thread
        async_result = pool.apply_async(solve_leetcode_problem, args=(test_name,))
        async_results += [(test_name, async_result)]
        # Fake async, to debug multiprocessing issues.
        # async_results += [(test_name, solve_leetcode_problem(test_name))]

    # Wait for all threads
    log.info("Waiting for tests to complete...")
    for test_name, async_result in async_results:
        try:
            results += [async_result.get()]
            # Fake async
            #results += [async_result]
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
