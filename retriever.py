from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
import chromadb

DOCUMENTS_PATH = 'documents'
CHROMA_PATH = './chroma'


embedding_model = SentenceTransformer('BAAI/bge-m3', local_files_only=False)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="my_collection", metadata={"hnsw:space": "ip"})


def load_documents():
    try:
        loader = DirectoryLoader(
            DOCUMENTS_PATH,
            glob='*.txt',
        )
        documents = loader.load()

        if not documents:
            raise ValueError(f"No documents found in {DOCUMENTS_PATH}")

        return documents
    except Exception as e:
        raise Exception(f"Error loading documents: {str(e)}")


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks created from documents")

    return chunks


def calculate_embeddings(chunks):
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    return embeddings


def save_to_chroma(embeddings, chunks):
    collection.add(
        ids = [f'chunk_{i}' for i in range(len(chunks))],
        embeddings = embeddings,
        documents = [chunk.page_content for chunk in chunks],
        metadatas = [chunk.metadata for chunk in chunks]

    )


def search(query: str, n_results: int = 5):
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    zipped_results = list(zip(results["documents"][0], results["metadatas"][0]))
    results_doc = [Document(page_content=doc[0], metadata=doc[1]) for doc in zipped_results]

    return results_doc


def build_vectorstore():
    if collection.count() > 0:
        print('vector store already exists')
        return
    documents = load_documents()
    chunks = split_text(documents)
    embeddings = calculate_embeddings(chunks)
    save_to_chroma(embeddings, chunks)



if __name__ == '__main__':
    build_vectorstore()


