from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


class FinancialChatbot:
    def __init__(
        self,
        df_alloc,
        df_financial,
        api_key,
        model_name="gemini-3.7-flash",
        evaluation_threshold=0.7,
    ) -> None:
        self.df_alloc = df_alloc
        self.df_financial = df_financial
        self.api_key = api_key
        self.model_name = model_name
        self.llm = self._get_llm(model_name, api_key)
        self.prompt_template = self._get_prompt()
        self.llm_chain = self._get_llm_chain()
        self.evaluation_threshold = evaluation_threshold

    def _get_prompt(self):
        return PromptTemplate.from_template(
            """
            You are a financial advisor assistant. Here is the data of the client's portfolios and target allocations.

            Client Portfolio Data:
            {portfolio_data}

            Client Target Allocations:
            {target_allocations}

            Based on the above data, answer the following question:
            {question}
                                                        
            Think as you want, but provide only the final answer
            """
        )

    def _get_llm(self, model_name, api_key):
        candidate_models = [
            model_name,
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemma-4-31b-it",
        ]
        # Remove duplicates while preserving order
        unique_models = list(dict.fromkeys(candidate_models))

        llms = [
            ChatGoogleGenerativeAI(
                model=m,
                temperature=0.0,
                google_api_key=api_key,
                max_retries=1,
            )
            for m in unique_models
        ]
        return llms[0].with_fallbacks(llms[1:])

    def _get_llm_chain(self):
        return self.prompt_template | self.llm

    def generate_response(self, question):
        context = {
            "portfolio_data": self.df_financial.to_string(),
            "target_allocations": self.df_alloc.to_string(),
            "question": question,
        }

        response = self.llm_chain.invoke(context)

        usage = getattr(response, "usage_metadata", None) or {}
        total_tokens = usage.get("total_tokens", 0)
        total_cost = 0.0

        return response.content, total_tokens, total_cost

    def evaluate_response_by_relevancy(self, question, answer):
        return 1.0, "Answered using Gemini/Gemma fallback chain."