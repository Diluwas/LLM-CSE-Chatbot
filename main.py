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
    # document_loader = Data_loader(configs)
    # document_loader.init_text_splitter()
    # splits = document_loader.load_and_process_pdfs()

    # Compute embeddings vectores and add to vector DB
    # vecrorDB.add_to_vector_db(splits)
    query = "What is stock market?"
    query_vector = ai_models.embedding_model.embed_query(query)
    # print("query_vector", query_vector, len(query_vector))

    results = vecrorDB.perform_similarity_search(query=query)
    # print("results =", results)
    context = "\n\n".join(doc.page_content for doc in results)
    # prompting
    prompt_manager = Prompt_Manager(configs.promt_template)
    prompt_manager.ask_question(context_data=context, question_data=query, llm=ai_models.llm_model)

if __name__ == "__main__":
    main()