class Configs:
    def __init__(self):
        self.azure_endpoint = "https://rag-models-dilun.openai.azure.com/"
        self.azure_llm_deployment = "gpt-4o"
        self.azure_embedding_model_deployment = "text-embedding-3-large"
        self.azure_openai_version = "2024-12-01-preview"
        self.text_embedding_model_name = "text-embedding-3-large"
        self.llm_model_name = "gpt-4o"
        self.mongo_client_endpoint = "mongodb://localhost:27017/"
        self.mongodb_collection = "rag_model_collection"
        self.index_name = "rag_vector_index"
        self.relevance_score_function = "cosine"
        self.pdf_loader_path = "./data"
        self.promt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}

        Question: {question}

        Helpful Answer:"""