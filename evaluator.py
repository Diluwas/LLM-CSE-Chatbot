#Context precision measures the signal-to-noise ratio of the retrieved context. This metric is computed using the question and the contexts.
#Context recall measures if all the relevant information required to answer the question was retrieved. This metric is computed based on the ground_truth (this is the only metric in the framework that relies on human-annotated ground truth labels) and the contexts.
#Faithfulness measures the factual accuracy of the generated answer. The number of correct statements from the given contexts is divided by the total number of statements in the generated answer. This metric uses the question, contextsand the answer.
#Answer relevancy measures how relevant the generated answer is to the question. This metric is computed using the question and the answer. For example, the answer “France is in western Europe.” to the question “Where is France and what is it’s capital?” would achieve a low answer relevancy because it only answers half of the question.

# {
#     "question": "What is the capital of France?",       # User Query
#     "contexts": ["Paris is the capital of France."],    # Retrieved Context
#     "answer": "The capital of France is Paris.",        # LLM generated answer
#     "ground_truth": "Paris"                             # What actual sources contain
# }

from models import Models 
from configs import Configs
from data_loader import Data_loader
from vectorDB import VectoreDB
from prompt_manager import Prompt_Manager
from datasets import Dataset
from ragas.metrics import faithfulness,answer_relevancy,context_recall,context_precision
from ragas import evaluate
import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

def main():

    configs = Configs()
    os.environ["AZURE_OPENAI_API_KEY"] = ""

    # other configuration
    azure_config = {
        "base_url": configs.azure_endpoint,  # your endpoint
        "model_deployment": configs.azure_llm_deployment,  # your model deployment name
        "model_name": configs.llm_model_name,  # your model name
        "embedding_deployment": configs.azure_embedding_model_deployment,  # your embedding deployment name
        "embedding_name": configs.text_embedding_model_name,  # your embedding name
    }

    evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_config["base_url"],
        azure_deployment=azure_config["model_deployment"],
        model=azure_config["model_name"],
        validate_base_url=False,
    ))

    # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_config["base_url"],
        azure_deployment=azure_config["embedding_deployment"],
        model=azure_config["embedding_name"],
    ))

    vecrorDB = VectoreDB(configs)

    #init required Ai models and vector DB
    ai_models = Models(configs)
    ai_models.init_llm_model()
    ai_models.init_embedding_model()

    # Compute embeddings vectores and add to vector DB            
    vecrorDB.load_vectordb_from_file(ai_models.embedding_model)
    
    data = [
        {
            "question": "What is the capital of France?",
            "contexts": [],
            "answer": "The capital of France is Paris.",
            "ground_truth": "Paris"
        },
        {
            "question": "dasdad",
            "contexts": [],
            "answer": "dasdad",
            "ground_truth": "dasd"
        }
    ]

    for data_set in data:
        print("Before: ", data_set)
        retrieved_results = vecrorDB.perform_similarity_search(query=data_set["question"])
        data_set["contexts"] = [doc.page_content for doc in retrieved_results]

        context = "\n\n".join(doc.page_content for doc in retrieved_results)

        prompt_manager = Prompt_Manager(configs.promt_template)
        response = prompt_manager.ask_question(context_data=context, question_data=data_set["question"], llm=ai_models.llm_model)
        data_set["answer"] = response.content

        print("After = ", data_set)

    dataset = Dataset.from_list(data)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    print(results)

            
if __name__ == "__main__":
    main()