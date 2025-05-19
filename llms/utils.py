"""This module contains utility functions for LLM operations."""

import os
from langchain_openai import AzureChatOpenAI

EAST_2_AZURE_ENDPOINT = os.environ.get("EAST_2_AZURE_ENDPOINT")
EAST_2_AZURE_API_KEY = os.environ.get("EAST_2_AZURE_API_KEY")


def get_4o_east_2_llm():
    return AzureChatOpenAI(
        azure_endpoint=EAST_2_AZURE_ENDPOINT,
        api_key=EAST_2_AZURE_API_KEY,
        deployment_name="gpt-4o-2",
        temperature=0,
        api_version="2024-08-01-preview",
    )


def get_o3_mini_east_2_llm(**kwargs) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=EAST_2_AZURE_ENDPOINT,
        api_key=EAST_2_AZURE_API_KEY,
        deployment_name="o3-mini",
        api_version="2025-01-01-preview",
        **kwargs
    )
