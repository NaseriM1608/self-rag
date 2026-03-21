import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(verbose=True)

llm = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0,
               api_key=os.getenv('GROQ_API_KEY'))