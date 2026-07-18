import os
import time
import logging

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please add it to your .env file."
            )

        self.client = Groq(api_key=api_key)

        self.model = "llama-3.3-70b-versatile"

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_retries: int = 3
    ):

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        for attempt in range(max_retries):
            try:

                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    messages=messages
                )

                return response.choices[0].message.content

            except Exception as e:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, e
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)