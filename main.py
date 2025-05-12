from models import Models 
from configs import Configs
from data_loader import Data_loader
from vectorDB import VectoreDB
from prompt_manager import Prompt_Manager

def main():
    configs = Configs()

    #init required Ai models and vector DB
    ai_models = Models(configs)
    ai_models.init_llm_model()
    ai_models.init_embedding_model()
    
    vecrorDB = VectoreDB(configs)
    vecrorDB.int_vectore_db(ai_models.embedding_model)

    #load documents and create chunks
    document_loader = Data_loader(configs)
    document_loader.init_text_splitter()
    splits = document_loader.load_and_process_pdfs()

    #Compute embeddings vectores and add to vector DB
    vecrorDB.add_to_vector_db(splits)

    #prompting
    prompt_manager = Prompt_Manager(configs.promt_template)
    prompt_manager.ask_question()

if __name__ == "__main__":
    main()