from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from config.prompts import FINAL_ANSWER_PROMPT


def assemble_final_answer(
    complex_query,
    retrieved_results,
    chat_model,
):

    context = "\n\n".join(
        [
            f"Sub-Query: {r['sub_query']}\n"
            f"Source: {r['best_source']}\n"
            f"Relevant Info: {r['best_chunk']}"
            for r in retrieved_results
        ]
    )

    messages = [
        SystemMessage(
            content=FINAL_ANSWER_PROMPT
        ),
        HumanMessage(
            content=
            f"Original Query: {complex_query}\n\n"
            f"Context:\n{context}\n\n"
            "Provide a comprehensive and fluent final answer."
        ),
    ]

    response = chat_model.invoke(messages)

    return response.content.strip()