from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

class Models:
    def __init__(self, configs):
        self.configs = configs
        self.llm_model = None
        self.embedding_model = None
        self.token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    
    def init_llm_model(self):
        self.llm_model = AzureChatOpenAI(
            azure_endpoint = self.configs.azure_endpoint,
            azure_deployment = self.configs.azure_llm_deployment,
            openai_api_version = self.configs.azure_openai_version,
            model = self.configs.llm_model_name,
            azure_ad_token_provider = self.token_provider
        )
    
    def init_embedding_model(self):
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint= self.configs.azure_endpoint,
            azure_deployment= self.configs.azure_embedding_model_deployment,
            openai_api_version= self.configs.azure_openai_version,
            model = self.configs.text_embedding_model_name,
            azure_ad_token_provider = self.token_provider
        )        