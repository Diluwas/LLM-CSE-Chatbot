from models import Models 
from configs import Configs
from data_loader import Data_loader
from vectorDB import VectoreDB
from prompt_manager import Prompt_Manager
import streamlit as st
import sys
import os

global Mode

def main():

    configs = Configs()
    vecrorDB = VectoreDB(configs)

    #init required Ai models and vector DB
    ai_models = Models(configs)
    ai_models.init_llm_model()
    ai_models.init_embedding_model()

    st.title("Stock Market Advisor")
    user_input = st.text_input("Enter your question about stock market:", "")
    if st.button("Submit"):
        try:
            if not os.path.exists(configs.db_file_path):
                print("Vector DB not found")
                # load documents and create chunks
                document_loader = Data_loader(configs)
                document_loader.init_text_splitter()
                splits = document_loader.load_and_process_pdfs()

                # Compute embeddings vectores and add to vector DB
                vecrorDB.int_vectore_db(ai_models.embedding_model)
                vecrorDB.add_to_vector_db(splits)
                vecrorDB.save_vector_db_to_the_file()
            
            else:
                print("Vector DB found.")
                vecrorDB.load_vectordb_from_file(ai_models.embedding_model)
            
            query_vector = ai_models.embedding_model.embed_query(user_input)
            results = vecrorDB.perform_similarity_search(query=user_input)
            context = "\n\n".join(doc.page_content for doc in results)     
            # prompting
            prompt_manager = Prompt_Manager(configs.promt_template)
            response = prompt_manager.ask_question(context_data=context, question_data=user_input, llm=ai_models.llm_model)
            st.write(response.content)

        except Exception as e:
            st.write(f"An error occurred: {e}")
            
if __name__ == "__main__":
    main()