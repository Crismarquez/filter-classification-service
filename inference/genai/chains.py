import uuid
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from inference.genai.retrieval import CognitiveSearch
from inference.genai.schemas import ClassificationOutput, get_classification_examples, get_simple_examples
from config.config import ENV_VARIABLES

class AssistantClassificator:
    def __init__(
        self,
        model_name: str = "gpt-4o"
    ):
        
        self.model_name = model_name
        self.cognitive_search = CognitiveSearch()

        self.llm_classification_service = ChatOpenAI(
    model=self.model_name,
    api_key=ENV_VARIABLES["OPENAI_KEY"],
    temperature=0.2
        )

    async def apredict(self, input_text):

        start = time.time()
        id_predict = str(uuid.uuid4())
        
        search_results = await self.cognitive_search.search(input_text, top=50)

        classification_examples = get_simple_examples(search_results)

        prompt = await self.classification_setup_prompt(
        )

        runnable = prompt | self.llm_classification_service.with_structured_output(
                schema=ClassificationOutput,
                method="function_calling",
                include_raw=False,
            )
        
        result = await runnable.ainvoke({"text_input": input_text,  "examples": classification_examples})

        classification_result = result.Classification
        explanation_result = result.Explanation

        #TODO: check if the classification is allowed

        return {
            "id_pred": id_predict,
            "result": classification_result,
            "metadata": {
                "time": time.time() - start,
                "explanation": explanation_result
            }
        }

    async def classification_setup_prompt(self) -> ChatPromptTemplate:
        """
        """

        system_prompt = f"""You are a classifier model designed to determine whether a message is spam or ham (non-spam). Your task is to read the incoming message and classify it into one of these two categories:

Spam: A message that is unsolicited, promotional, or attempting to deceive the recipient.
Ham: A legitimate message that does not have the characteristics of spam.

For each message, provide the following:
- The classification (spam or ham).
- A short explanation justifying why the message fits the category you selected. Use clear and concise reasoning based on features such as:
        Unsolicited promotional content.
        Use of financial incentives, prizes, or promotions.
        Presence of suspicious links or requests for personal information.
        Informal or irrelevant content typical of personal or legitimate communications.

Here are a few examples of your responses:

Example 1:
Message: "Free entry in a competition to win £1000! Text WIN to 12345 now."
Classification: spam.
Explanation: The message promotes a contest with financial incentives and includes a request for the recipient to take an action (text WIN), which is typical of spam.

Example 2:
Message: "Hey, are we still meeting for lunch tomorrow?"
Classification: ham.
Explanation: This is a personal message without any promotional content or suspicious elements, typical of legitimate communication.

Instructions: After classifying each message, always provide a clear and relevant justification based on the message’s content."""
        
        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
        ),
        #MessagesPlaceholder("examples"),
        ("human", "# Some classifications examples: {examples} \n\n new message: {text_input}"),
    ]
)
        
        return prompt
