import logging
import time
from pprint import pformat

import openai
from pydantic import BaseModel
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

client = openai.Client()


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
