from typing import Dict, Optional, List, Tuple
import json
import time
import asyncio
import uuid

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import tiktoken

from config.prompt import get_selfquery_examples, systemprompt_multiquery, systemprompt_selfquery, systemprompt_synthesis, contextualize_q_system_prompt, systemprompt_followup_q
from assistant.retriever import CognitiveSearch
from assistant.schemas import Multiquery, Queries, Questions, Condensor
from assistant.utils import delete_content_duplicates, format_docs
from config.config import ENV_VARIABLES, CONFIG_DIR, logger

class RAG:
    def __init__(
        self,
        retrieval_args: Optional[Dict] = None,
            ) -> None:
        """
        Initializes an instance of the class.

        Args:

        Returns:
            None: This function does not return anything.
        """

        self.retrieval_args = retrieval_args
        self.cognitive_search = CognitiveSearch()

        if self.retrieval_args is None:
            self.retrieval_args = {}

        self.k_top = self.retrieval_args.get("k_top", 8)

        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4o", #"gpt-35-turbo-16k",#"gpt-4o",
    api_key=ENV_VARIABLES["AZURE_OPENAI_API_KEY"],
    azure_endpoint=f"https://{ENV_VARIABLES['AZURE_OPENAI_SERVICE']}.openai.azure.com/",
    api_version="2024-04-01-preview",
    temperature=0.5,
    seed=42,
)
        self.llm_followup_q = AzureChatOpenAI(
            deployment_name="gpt-4o-mini", #"gpt-35-turbo-16k",#"gpt-4o",
    api_key=ENV_VARIABLES["AZURE_OPENAI_API_KEY"],
    azure_endpoint=f"https://{ENV_VARIABLES['AZURE_OPENAI_SERVICE']}.openai.azure.com/",
    api_version="2024-04-01-preview",
    temperature=0.8,
    seed=42,
)

        self.llm_synthesis = AzureChatOpenAI(
            deployment_name="gpt-4o", #"gpt-35-turbo-16k",#"gpt-4o",
    api_key=ENV_VARIABLES["AZURE_OPENAI_API_KEY"],
    azure_endpoint=f"https://{ENV_VARIABLES['AZURE_OPENAI_SERVICE']}.openai.azure.com/",
    api_version="2024-04-01-preview",
    temperature=0.8,
    seed=42,
)

        self.encoding = tiktoken.encoding_for_model("gpt-4")

    async def stream_arun(self, input: Dict):
        message_id = str(uuid.uuid4())
        start = time.time()

        input_user = history[-1]["content"]
        history = await self.format_prompt_history(history)

        # condensor
        time_condensor = time.time()
        last_query = input_user
        prompt = await self.contextualize_prompt()
        runnable = prompt | self.llm.with_structured_output(schema=Condensor, method="function_calling", include_raw=False)
        response = await runnable.ainvoke({"input": last_query, "chat_history": history})
        time_condensor = time.time() - time_condensor
        print(f"Time condensor: {time_condensor}")

        user_query = response.condensor
        topic = response.topic

        #TODO reflection node

        # multy-query
        time_multiquery = time.time()
        prompt = await self.multiquery_setup_prompt()
        runnable = prompt | self.llm.with_structured_output(schema=Multiquery, method="function_calling", include_raw=False)
        time_multiquery = time.time() - time_multiquery
        response = await runnable.ainvoke({"input_user": user_query, "examples": []})

        list_multiqueries = response.queries
        list_multiqueries = [{"query": query.query, "domain": topic} for query in list_multiqueries]


        # search - embeddings
        time_search = time.time()
        #documents = await self.parallelize(list_multiqueries)
        documents = []
        time_search = time.time() - time_search
        print(f"Time search: {time_search}")

        # reduce - format
        #unique_documents = await delete_content_duplicates(documents)
        unique_documents = []
        references = [{"link": doc["link"], "name_to_show": f'{doc["country"]}_{doc["year"]}'} for doc in unique_documents if doc["link"] != ""]

        context = await format_docs(unique_documents)

        id_content = [doc["id_content"] for doc in unique_documents]

        # followup questions
        time_followup_q = time.time()
        prompt_followup_q = await self.followupquestion_setup_prompt()
        runnable_followup_q = prompt_followup_q | self.llm_followup_q.with_structured_output(schema=Questions, method="function_calling", include_raw=False)
        followup_response = await runnable_followup_q.ainvoke({"input_user": user_query, "context": context})
        list_followup_questions = [{"question": followup_q.followup_question, "question_to_show": followup_q.followup_question_summary} for followup_q in followup_response.followup_questions]
        time_followup_q = time.time() - time_followup_q

        # Synthesis
        time_synthesis = time.time()
        prompt_synthesis = await self.synthesis_setup_prompt()
        runnable_synthesis = prompt_synthesis | self.llm_synthesis
        #synthesis_response = await runnable_synthesis.ainvoke({"input_user": user_query, "context": context})
        time_synthesis = time.time() - time_synthesis

        #response_content = synthesis_response.content

        chain_logs = {
                "time_condensor": time_condensor,
                "time_multiquery": time_multiquery,
                "time_search": time_search,
                "time_total": time.time() - start,
                "retrieval": {
                    "id_content": id_content
                },
                "node_condensor": {
                    "query": user_query
                },
                "node_multyquery": {
                    "queries": list_multiqueries
                }
        }

        async for event in runnable_synthesis.astream_events(
                {"input_user": user_query, "context": context},
                version="v1",
            ):
                event["message_id"] = message_id
                event["followup_questions"] = list_followup_questions
                event["chain_logs"] = chain_logs
                yield event

    async def a_run(self, input: Dict, debug=False) -> Dict:
        """
        Runs prompts and returns the response.

        Parameters:
        - input: A dictionary containing the input data for the function.
        Returns:
        - response_dict: A dictionary containing the response data.
        """
        
        message_id = str(uuid.uuid4())
        tokens_used = {}
        start = time.time()

        conversation_id = input.get("conversation_id", None)
        history = input.get("history", None)
        #logger.info(f"HISTORY: {history}")
        input_user = history[-1]["content"]


        # condensor
        time_condensor = time.time()
        last_query = input_user
        prompt = await self.contextualize_prompt()
        runnable = prompt | self.llm.with_structured_output(schema=Condensor, method="function_calling", include_raw=False)
        response = await runnable.ainvoke({"input": last_query, "chat_history": history})
        time_condensor = time.time() - time_condensor
        print(f"Time condensor: {time_condensor}")

        user_query = response.condensor
        topic = response.topic

        #TODO reflection node

        # multy-query
        time_multiquery = time.time()
        prompt = await self.multiquery_setup_prompt()
        runnable = prompt | self.llm.with_structured_output(schema=Multiquery, method="function_calling", include_raw=False)
        time_multiquery = time.time() - time_multiquery
        response = await runnable.ainvoke({"input_user": user_query, "examples": []})

        list_multiqueries = response.queries
        list_multiqueries = [{"query": query.query, "domain": topic} for query in list_multiqueries]


        # search - embeddings
        time_search = time.time()
        #documents = await self.parallelize(list_multiqueries)
        documents = []
        time_search = time.time() - time_search
        print(f"Time search: {time_search}")

        # reduce - format
        #unique_documents = await delete_content_duplicates(documents)
        unique_documents = []
        references = [{"link": doc["link"], "name_to_show": f'{doc["country"]}_{doc["year"]}'} for doc in unique_documents if doc["link"] != ""]

        context = await format_docs(unique_documents)

        id_content = [doc["id_content"] for doc in unique_documents]

        # followup questions
        time_followup_q = time.time()
        prompt_followup_q = await self.followupquestion_setup_prompt()
        runnable_followup_q = prompt_followup_q | self.llm_followup_q.with_structured_output(schema=Questions, method="function_calling", include_raw=False)
        followup_response = await runnable_followup_q.ainvoke({"input_user": user_query, "context": context})
        list_followup_questions = [{"question": followup_q.followup_question, "question_to_show": followup_q.followup_question_summary} for followup_q in followup_response.followup_questions]
        time_followup_q = time.time() - time_followup_q

        # Synthesis
        time_synthesis = time.time()
        prompt_synthesis = await self.synthesis_setup_prompt()
        runnable_synthesis = prompt_synthesis | self.llm_synthesis
        synthesis_response = await runnable_synthesis.ainvoke({"input_user": user_query, "context": context})
        time_synthesis = time.time() - time_synthesis

        response_content = synthesis_response.content

        chain_logs = {
                "time_condensor": time_condensor,
                "time_multiquery": time_multiquery,
                "time_search": time_search,
                "time_total": time.time() - start,
                "retrieval": {
                    "id_content": id_content
                },
                "node_condensor": {
                    "query": user_query
                },
                "node_multyquery": {
                    "queries": list_multiqueries
                },
                "node_synthesis": {
                    "response": response_content,
                }
        }

        if debug:
            return {
                "response": response_content, 
                "references": references,
                "followup_questions":list_followup_questions,
                "metadata": chain_logs
            }
        else:
            return {
                "response": response_content,
                "references": references,
                "followup_questions": list_followup_questions,
                }
    

    async def parallelize(self, list_query: List[Dict]) -> List[str]:
        tasks = [self.cognitive_search.search_with_filters(
            text_query=query["query"], year_filter=query["year"], country_filter=query["country"], region_filter=query["region"], top=self.k_top) for query in list_query]
        all_documents = await asyncio.gather(*tasks)
        # all_documents = []
        # for query in list_query:
        #     documents = await self.cognitive_search.search_with_filters(exit(9)
        #         text_query = query.query,
        #         year_filter=query.year, 
        #         country_filter=query.country
        #         )
        #     all_documents.append(documents)
        return all_documents

    async def format_prompt_history(self, history: List[Dict]) -> List:
        chat_history = []
        for message in history[:-1]:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                chat_history.append(AIMessage(content=message["content"]))
            elif message["role"] == "system":
                chat_history.append(SystemMessage(content=message["content"]))
            else:
                raise ValueError(f"Unknown message role: {message['role']}")
        return chat_history

    async def synthesis_setup_prompt(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            systemprompt_synthesis
        ),
        #MessagesPlaceholder("examples"),
        ("human",  """
         user_question: {input_user}

        # Context : {context}

Please respond in this language: {language}
         """),
    ]
)
        return prompt
    

    async def multiquery_setup_prompt(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            systemprompt_multiquery
        ),
        MessagesPlaceholder("examples"),
        ("human",  "{input_user}"),
    ]
)  
        return prompt

    async def contextualize_prompt(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            contextualize_q_system_prompt
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",  "last user question: {input}"),
    ]
)  
        return prompt
    
    async def followupquestion_setup_prompt(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            systemprompt_followup_q
        ),
        #MessagesPlaceholder("examples"),
        ("human",  "#Context: {context} \n User question: {input_user}"),
    ]
)
        return prompt