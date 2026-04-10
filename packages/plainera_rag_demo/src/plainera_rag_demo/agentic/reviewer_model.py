from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True, slots=True)
class OpenAIReviewerModel:
    """Minimal Responses API adapter for the prompted grounding reviewer."""

    client: OpenAI
    model: str

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text output for the bounded reviewer prompt."""
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
        )

        text = response.output_text
        if not text:
            raise ValueError("Reviewer model returned empty output_text.")

        return text
