import logging
import os
import subprocess
import time
import uuid
from contextlib import redirect_stdout
from io import StringIO

import pytest
from pydantic import BaseModel
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

BASE_TESTDIR = "/tmp/leetcode-bot"


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

        # Clear and re-create the output directory.
        os.system(f"rm -rf {BASE_TESTDIR}")
        os.system(f"mkdir -p {BASE_TESTDIR}")

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
        subprocess.check_call(["rm", "-rf", TESTDIR])
        subprocess.check_call(["mkdir", "-p", TESTDIR])
        # Copy the utils/ tree into TESTDIR
        subprocess.check_call(["cp", "-r", f"{os.path.dirname(__file__)}/leetcode/utils", f"{TESTDIR}/"])

        with open(tmp_testfile, "w") as file:
            file.write(test_file_with_candidate)

        # Run pytest and capture the exit code + result to a str.
        log.info(f"Running test {self._test_name}...")
        temp_stdout = StringIO()
        with redirect_stdout(temp_stdout):
            self._exit_code = pytest.main(["-v", tmp_testfile, f"--rootdir={TESTDIR}"])
            self._log_lines = temp_stdout.getvalue()

        # Run using subprocess
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
