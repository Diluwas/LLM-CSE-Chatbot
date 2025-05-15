import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

class VectoreDB:
    def __init__(self, configs):
        self.vector_store = None
        self.configs = configs
        self.index = faiss.IndexHNSWFlat(self.configs.embedding_length, 32)

    def int_vectore_db(self, embedding_function):
        self.vector_store = FAISS(
            embedding_function = embedding_function,
            index = self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_to_vector_db(self, splitted_docs):
        self.vector_store.add_documents(splitted_docs)

    def perform_similarity_search(self, query):
        results = self.vector_store.similarity_search(query=query, k=5)
        return results
    
    def save_vector_db_to_the_file(self):
        self.vector_store.save_local(self.configs.db_file_path)

    def load_vectordb_from_file(self, embedding_function):
        new_vector_store = FAISS.load_local(self.configs.db_file_path, embedding_function, allow_dangerous_deserialization=True)
        self.vector_store = new_vector_store
