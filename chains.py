"""Prompt templates and LCEL chains.

Graders use structured outputs (Pydantic models) instead of YES/NO string
parsing. Generation is exposed as a factory so the graph can escalate
temperature across grounding-failure retries.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from llm import llm, make_llm


class RelevanceVerdict(BaseModel):
    """Whether a single retrieved chunk helps answer the question."""

    is_relevant: bool = Field(
        description="true only if the chunk clearly helps answer the question"
    )


class GroundingVerdict(BaseModel):
    """Whether every claim in the generated answer is supported by the docs."""

    is_grounded: bool = Field(
        description="true only if every key claim in the answer is supported "
        "by the provided documents"
    )


class UsefulnessVerdict(BaseModel):
    """Whether the generated answer actually addresses the question."""

    is_useful: bool = Field(
        description="true only if the answer directly addresses the question"
    )


# 1. Relevance grading
relevance_prompt = ChatPromptTemplate.from_template(
    """
    You are a relevance grader in a retrieval-augmented system.

    Determine whether the document chunk is useful for answering the question.

    Mark is_relevant=true if:
    - The chunk contains information that directly helps answer the question
    - The chunk includes partial but useful information related to the question
    - The chunk contains keywords or concepts clearly connected to the question

    Mark is_relevant=false if:
    - The chunk is unrelated to the question
    - The chunk is too vague, generic, or off-topic

    Be strict. Only mark true if the chunk has clear relevance; if in doubt,
    mark false.

    User Question:
    {question}

    Document Chunk:
    {chunk}
    """
)

relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceVerdict, method="function_calling")


# 2. Generation
generation_prompt = ChatPromptTemplate.from_template(
    """
    You are a question-answering system that must answer using only the
    retrieved documents provided in the prompt.

    Instructions:
    - Use only the information contained in those documents.
    - Do not use outside knowledge.
    - Do not guess or invent missing details.
    - If the answer cannot be found in the documents, reply exactly:
      "I don't know based on the provided documents."
    - Be concise, but include all important information that is supported
      by the documents.
    - If multiple documents are relevant, combine their information into
      one answer.

    Citation rules:
    - Every factual statement taken from a document must be cited immediately
      after the statement.
    - Each document is labeled like [1], [2], etc., and has a source
      name/title after the label.
    - Cite using this format: [1: Source Name]
    - If more than one document supports the same statement, include all
      relevant citations.
    - Do not invent, modify, or assume any source identifiers or titles.
    - Only cite identifiers and titles that appear in the provided documents.

    User Question:
    {question}

    Documents:
    {documents}
    """
)


def make_generation_chain(temperature: float) -> Runnable:
    """Generation chain at an explicit temperature (retries escalate it)."""
    return generation_prompt | make_llm(temperature) | StrOutputParser()


# 3. Grounding check
grounding_prompt = ChatPromptTemplate.from_template(
    """
    You are a grounding verifier in a retrieval-augmented system.

    Determine whether the generated answer is fully supported by the provided
    documents.

    Mark is_grounded=true if:
    - Every key claim in the answer is supported by the documents
    - The answer does not include information outside the documents
    - The answer does not contradict the documents

    Mark is_grounded=false if:
    - Any part of the answer is not supported by the documents
    - The answer includes additional information not found in the documents
    - The answer contradicts the documents

    Be strict about claims stated as facts that are not in the documents, but
    treat valid logical, numeric, temporal, or transitive inferences drawn
    from the documents as grounded.

    Generated Answer:
    {answer}

    Documents:
    {documents}
    """
)

grounding_chain = grounding_prompt | llm.with_structured_output(GroundingVerdict, method="function_calling")


# 4. Usefulness check
usefulness_prompt = ChatPromptTemplate.from_template(
    """
    You are a usefulness evaluator in a question-answering system.

    Determine whether the generated answer actually addresses the question.

    Mark is_useful=true if:
    - The answer directly addresses the question
    - The answer is specific and relevant to what was asked
    - The answer provides a clear and meaningful response

    Mark is_useful=false if:
    - The answer is vague, generic, or evasive
    - The answer does not address the actual question
    - The answer is off-topic or only partially relevant

    Be strict. An answer that sounds good but does not truly answer the
    question should be marked false. Focus only on the relationship between
    the question and the answer.

    User Question:
    {question}

    Generated Answer:
    {answer}
    """
)

usefulness_chain = usefulness_prompt | llm.with_structured_output(UsefulnessVerdict, method="function_calling")
