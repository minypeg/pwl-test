"""This module contains the nodes of the graph."""

from llms import (
    medical_analyzer_chain,
)
import aiofiles


async def general_summarizer(state):
    """
    Summarize the documents

    Args:
        state (dict): The current graph state
    """

    async with aiofiles.open("text.txt", "r") as f:
        text = await f.read()

    result = await medical_analyzer_chain.ainvoke(
        {
            "documents": text,
        }
    )

    return {
        "final_summary": result.model_dump(),
    }
