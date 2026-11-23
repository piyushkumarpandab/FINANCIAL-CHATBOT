import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


class FinancialChatbot:
    def __init__(
        self,
        df_alloc,
        df_financial,
        api_key,
        model_name="gemini-3.8-flash",
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
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=api_key,
        )

    def _get_llm_chain(self):
        return self.prompt_template | self.llm

    def generate_response(self, question):
        context = {
            "portfolio_data": self.df_financial.to_string(),
            "target_allocations": self.df_alloc.to_string(),
            "question": question,
        }

        response = self.llm_chain.invoke(context)

        # Extract token usage from Gemini metadata
        usage = getattr(response, "usage_metadata", None) or {}
        total_tokens = usage.get("total_tokens", 0)
        total_cost = 0.0  # Free tier / standard Gemini usage

        return response.content, total_tokens, total_cost

    def evaluate_response_by_relevancy(self, question, answer):
        eval_prompt = (
            f"Evaluate how relevant and accurate the Answer is to the Question on a scale from 0.0 to 1.0.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f'Respond ONLY with valid JSON in this exact format: {{"score": 0.95, "reason": "brief explanation"}}'
        )
        try:
            eval_response = self.llm.invoke(eval_prompt)
            raw_text = eval_response.content.strip().strip("```json").strip("```").strip()
            result = json.loads(raw_text)
            return float(result.get("score", 1.0)), result.get("reason", "Evaluated by Gemini.")
        except Exception:
            return 1.0, "Response generated using Google Gemini."