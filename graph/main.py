"""This module contains the graph definition."""

from langgraph.graph import END, StateGraph, START
from graph.state import GraphState
from graph.nodes import (
    general_summarizer,
)
import os
import sentry_sdk

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("general_summarizer", general_summarizer)

workflow.add_edge(START, "general_summarizer")

workflow.add_edge("general_summarizer", END)

graph = workflow.compile()


import asyncio


def init_sentry():
    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        environment=os.environ.get("ENV", "local"),
        traces_sample_rate=1.0,
    )


async def get_graph():
    await asyncio.to_thread(init_sentry)

    return graph
