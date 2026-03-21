from langchain_core.prompts import ChatPromptTemplate


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


# 3. Generation prompt
generation_prompt = ChatPromptTemplate.from_template(
    """
    You are a question-answering system using retrieved documents.

    Your task is to answer the user’s question using ONLY the provided documents.
    
    Rules:
    - Use only the information from the documents
    - Do NOT use prior knowledge or make up information
    - If the answer is not contained in the documents, say:
      "I don't know based on the provided documents."
    - Be concise but complete
    - If multiple documents are relevant, combine their information
    
    Citations:
    - Cite sources for every key statement
    - Use the format: [source_id]
    - Each document will include a source identifier (e.g., [1], [2])
    - Place citations immediately after the relevant information
    - If multiple sources support a statement, include multiple citations (e.g., [1][3])
    
    Output:
    - Provide a clear, direct answer to the question
    - Include citations inline as specified
    - Do not mention the documents explicitly (e.g., do not say "according to the documents")
    
    User Question:
    {question}
    
    Documents:
    {documents}
    """
)


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