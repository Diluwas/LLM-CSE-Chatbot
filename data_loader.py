from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

class Data_loader:
    def __init__(self, configs):
        self.configs = configs
        self.text_splitter = None

    def init_text_splitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            add_start_index=True,
        )

    def load_and_process_pdfs(self):
        documents = []
        for file in os.listdir(self.configs.pdf_loader_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(self.configs.pdf_loader_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
        splits = self.text_splitter.split_documents(documents)
        print(f"Split pdf into {len(splits)} sub-documents.")
        return splits
        

