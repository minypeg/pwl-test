"""This module contains the state of the graph."""

from typing_extensions import TypedDict
from llms.medical_analyzer import FinalAnalysis


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        summarized_documents: list of summarized documents
    """

    final_summary: FinalAnalysis = None
