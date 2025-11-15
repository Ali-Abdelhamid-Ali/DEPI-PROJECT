from .BaseController import BaseController 
from stores.llm.LLMProviderFactory import LLMProviderFactory
from helpers.history import load_history

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.language_models import LLM
from pydantic import Field

class ProviderLLMWrapper(LLM):
    provider: any = Field(..., description="The wrapped LLM provider")
    def _call(self, prompt: str, stop=None):
        return self.provider.generate_text(prompt)
    @property
    def _llm_type(self):
        return "custom-provider"
    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

class ChatController(BaseController):
    def __init__(self, session_id=None,username=None,chat_history=None,utility_params=None): 
        super().__init__() 
        self.session_id = session_id
        self.username = username
        self.chat_history=chat_history or []
        self.utility_params=utility_params

    async def completion_router(self, user_prompt: str):
        completion_type = self.utility_params.get("completion_type")
        if completion_type == "chat":
            return await self.main_chain(user_prompt)
        elif completion_type == "main":
            return await self.main_chain_with_history(user_prompt)
        else:
            raise ValueError("Unsupported completion type")


    async def main_chain(self, user_prompt):
        app_settings = self.app_settings
        factory = LLMProviderFactory(config=app_settings)
        provider = factory.create(provider=app_settings.GENERATION_BACKEND)
        provider_runnable = RunnableLambda(
            lambda chat_value: provider.generate_text(chat_value.to_string())
        )
        template = ChatPromptTemplate.from_template(
            "Answer as if you were the President of the United States: {prompt}"
        )
        chain = template | provider_runnable
        result = chain.invoke({"prompt": user_prompt})
        return result

    async def main_chain_with_history(self, user_prompt, max_summary_tokens=500):
        app_settings = self.app_settings
        factory = LLMProviderFactory(config=app_settings)
        provider = factory.create(provider=app_settings.GENERATION_BACKEND)
        lc_llm = ProviderLLMWrapper(provider=provider)

        full_history = await load_history(self.session_id)

        n_recent_messages = 5
        last_msgs = full_history[-n_recent_messages:]
        older_msgs = full_history[:-n_recent_messages]

        # for msg in last_msgs:
        #     print(msg)
        #     print("**************")
        # print("=============================================")
        # for msg in older_msgs:
        #     print(msg)
        #     print("**************")
        # print("=============================================")
        
        recent_memory = ConversationBufferMemory(
            memory_key="recent",
            return_messages=True
        )
        for msg in reversed(last_msgs):
            recent_memory.save_context({"input": msg[0].content}, {"output": msg[1].content})

        recent_text = recent_memory.load_memory_variables({})["recent"]

        summary_memory = ConversationSummaryBufferMemory(
            llm=lc_llm,
            memory_key="summary",
            return_messages=False
        )

        token_count = 0
        for msg in reversed(older_msgs):
            user_tokens=lc_llm.get_num_tokens(msg[0].content)
            ai_tokens=lc_llm.get_num_tokens(msg[1].content)
            msg_tokens = user_tokens+ai_tokens
            if token_count + msg_tokens > max_summary_tokens:
                break 
            summary_memory.save_context({"input": msg[0].content}, {"output": msg[1].content})
            token_count += msg_tokens
        summary_text = summary_memory.load_memory_variables({})["summary"]

        template = ChatPromptTemplate.from_template("""
            answer this user prompt: {prompt}

            and these are the Recent Messages between you and the user:
            {recent}
            
            with the help of this Conversation Summary:
            {summary}                                             
        """)

        model_inputs = {
            "summary": summary_text,
            "recent": recent_text,
            "prompt": user_prompt
        }

        # prompt_value = template.format_prompt(**model_inputs)
        # print(prompt_value.to_string())
        
        provider_runnable = RunnableLambda(
            lambda v: provider.generate_text(v.to_string())
        )

        chain = template | provider_runnable
        response = chain.invoke(model_inputs)
        return response

    
 