from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm import llm


# 1. Retrieval decision
retrieval_prompt = ChatPromptTemplate.from_template(
    """
    You are a decision system for a retrieval-augmented AI.

    Your task is to determine whether external documents are required to answer the user’s question.
    
    Rules:
    - Answer "YES" if:
      - The question requires up-to-date, real-time, or changing information
      - The question depends on specific documents, proprietary data, or unknown context
      - The answer requires high factual precision and cannot rely on general knowledge
      - You are uncertain about the answer
    
    - Answer "NO" if:
      - The question can be answered using general knowledge
      - The question is conceptual, explanatory, or common knowledge
      - The answer does not depend on external or private data
    
    Output format:
    - Respond with only one word:
      - YES
      - NO
    - Do not explain your answer.
    
    User Question:
    {question}
    """
)

retrieval_chain = retrieval_prompt | llm | StrOutputParser()


# 2. Relevance prompt
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


# 3. Generation prompt
generation_prompt = ChatPromptTemplate.from_template(
    """
    You are a question-answering system using retrieved documents.

    Your task is to answer the user’s question using the provided documents when they are available.
    
    Rules:
    - If documents are provided:
      - Use only the information from the documents
      - Do NOT use prior knowledge or make up information beyond the documents
      - If the answer is not contained in the documents, say:
        "I don't know based on the provided documents."
    
    - If NO documents are provided:
      - Answer using your own knowledge
    
    - Be concise but complete
    - If multiple documents are relevant, combine their information
    
    Citations:
    - If documents are provided:
      - Cite sources for every key statement
      - Use the format: [source_id]
      - Each document will include a source identifier (e.g., [1], [2])
      - Place citations immediately after the relevant information
      - If multiple sources support a statement, include multiple citations (e.g., [1][3])
    
    - If NO documents are provided:
      - Do NOT include citations
    
    Output:
    - Provide a clear, direct answer to the question
    - Follow the citation rules depending on whether documents are provided
    - Do not mention the documents explicitly (e.g., do not say "according to the documents")
    
    User Question:
    {question}
    
    Documents:
    {documents}
    """
)

generation_chain = generation_prompt | llm | StrOutputParser()


# 4. Grounding check
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


# 5. Usefulness check
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