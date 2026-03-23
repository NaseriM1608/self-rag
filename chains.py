from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm import llm


# 1. Relevance prompt
relevance_prompt = ChatPromptTemplate.from_template(
    """
    You are a relevance grader in a retrieval-augmented system.

    Your task is to determine whether the given document chunk is useful for answering the user’s question.
    
    Rules:
    - Answer "YES" if:
      - The chunk contains information that directly helps answer the question
      - The chunk includes partial but useful information related to the question
      - The chunk contains keywords or concepts clearly connected to the question
    
    - Answer "NO" if:
      - The chunk is unrelated to the question
      - The chunk is too vague, generic, or off-topic
      - The chunk does not provide any useful information for answering the question
    
    Important:
    - Be strict. Only mark "YES" if the chunk has clear relevance.
    - If in doubt, answer "NO".
    
    Output format:
    - Respond with only one word:
      - YES
      - NO
    - Do not explain your answer.
    
    User Question:
    {question}
    
    Document Chunk:
    {chunk}
    """
)

relevance_chain = relevance_prompt | llm | StrOutputParser()


# 2. Generation prompt
generation_prompt = ChatPromptTemplate.from_template(
    """
    You are a question-answering system that must answer using only the retrieved documents provided in the prompt.

    Instructions:
    - If documents are provided:
      - Use only the information contained in those documents.
      - Do not use outside knowledge.
      - Do not guess or invent missing details.
      - If the answer cannot be found in the documents, reply exactly: "I don't know based on the provided documents."
    - If no documents are provided, or the documents field is empty, reply exactly: "No relevant documents were found."
    - Be concise, but include all important information that is supported by the documents.
    - If multiple documents are relevant, combine their information into one answer.
    
    Citation rules:
    - Every factual statement taken from a document must be cited immediately after the statement.
    - Each document is labeled like [1], [2], etc., and has a source name/title after the label.
    - Cite using this format: [1: Source Name]
    - If more than one document supports the same statement, include all relevant citations.
    - Do not invent, modify, or assume any source identifiers or titles.
    - Only cite identifiers and titles that appear in the provided documents.
    
    User Question:
    {question}
    
    Documents:
    {documents}
    """
)

generation_chain = generation_prompt | llm | StrOutputParser()


# 3. Grounding check
grounding_prompt = ChatPromptTemplate.from_template(
    """
    You are a grounding verifier in a retrieval-augmented system.

    Your task is to determine whether the generated answer is fully supported by the provided documents.
    
    Rules:
    - Answer "YES" if:
      - Every key claim in the answer is supported by the documents
      - The answer does not include information outside the documents
      - The answer does not contradict the documents
    
    - Answer "NO" if:
      - Any part of the answer is not supported by the documents
      - The answer includes additional information not found in the documents
      - The answer contradicts the documents
    
    Important:
    - Be strict. Even small unsupported additions should result in "NO".
    - Do not assume facts not explicitly stated in the documents.
    
    Output format:
    - Respond with only one word:
      - YES
      - NO
    - Do not explain your answer.
    
    Generated Answer:
    {answer}
    
    Documents:
    {documents}
    """
)

grounding_chain = grounding_prompt | llm | StrOutputParser()


# 4. Usefulness check
usefulness_prompt = ChatPromptTemplate.from_template(
    """
    You are a usefulness evaluator in a question-answering system.

    Your task is to determine whether the generated answer actually addresses the user’s question.
    
    Rules:
    - Answer "YES" if:
      - The answer directly addresses the question
      - The answer is specific and relevant to what was asked
      - The answer provides a clear and meaningful response
    
    - Answer "NO" if:
      - The answer is vague, generic, or evasive
      - The answer does not address the actual question
      - The answer is off-topic or only partially relevant
      - The answer avoids giving a clear response
    
    Important:
    - Be strict. An answer that sounds good but does not truly answer the question should be marked "NO".
    - Focus only on the relationship between the question and the answer.
    
    Output format:
    - Respond with only one word:
      - YES
      - NO
    - Do not explain your answer.
    
    User Question:
    {question}
    
    Generated Answer:
    {answer}
    """
)

usefulness_chain = usefulness_prompt | llm | StrOutputParser()