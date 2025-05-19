"""This module contains utility functions for LLM operations."""

import os
from langchain_openai import ChatOpenAI

EAST_2_AZURE_ENDPOINT = os.environ.get("EAST_2_AZURE_ENDPOINT")
EAST_2_AZURE_API_KEY = os.environ.get("EAST_2_AZURE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_o3_mini_east_2_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="o3-mini",
        disabled_params={"parallel_tool_calls": None},
    )
