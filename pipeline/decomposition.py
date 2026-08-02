import json

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from config.prompts import QUERY_DECOMPOSITION_PROMPT


def decompose_query(complex_query, chat_model):
    """
    Break a complex query into multiple simpler sub-queries.
    """

    messages = [
        SystemMessage(content=QUERY_DECOMPOSITION_PROMPT),
        HumanMessage(
            content=f"Complex Query: {complex_query}"
        ),
    ]

    response = chat_model.invoke(messages)

    raw = (
        response.content.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return json.loads(raw)