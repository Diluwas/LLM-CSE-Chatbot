from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

class VectoreDB:
    def __init__(self, configs):
        self.vector_store = None
        self.configs = configs
        self.client = MongoClient(configs.mongo_client_endpoint)
        self.db = self.client["rag_database"]
        self.collection = self.db[configs.mongodb_collection]

    def int_vectore_db(self, embedding_function):
        self.vector_store = MongoDBAtlasVectorSearch(
            embedding = embedding_function,
            collection = self.collection,
            index_name = self.configs.index_name,
            relevance_score_fn = self.configs.relevance_score_function,
        )

    def add_to_vector_db(self, splitted_docs):
        self.vector_store.add_documents(splitted_docs)
