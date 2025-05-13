from langchain import hub
from langchain_core.prompts import PromptTemplate

class Prompt_Manager():
    def __init__(self, template):
        self.custom_rag_prompt = PromptTemplate.from_template(template)

    def ask_question(self, context_data, question_data, llm):
        formatted_prompt = self.custom_rag_prompt.format(context=context_data, question=question_data)
        response = llm.invoke(formatted_prompt)
        print(f"Response = {response.content}")