import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from inference.genai.schemas import ClassificationOutput
from config.prompt import image_analysis_prompt
from config.config import ENV_VARIABLES

class ImageAnalyser:
    def __init__(self, model_type: str = "gpt-4o"):
        self.model_type = model_type
        self.model = ChatOpenAI(
    model=model_type,
    api_key=ENV_VARIABLES["OPENAI_KEY"],
    temperature=0.2
        )

    async def apredict(self, image_base64):

        if image_base64 is None:
            return {}

        id_predict = str(uuid.uuid4())
        imgs_content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]

        structured_model = self.model.with_structured_output(
            schema=ClassificationOutput,
            method="function_calling",
            include_raw=False,
        )

        result = await structured_model.ainvoke(
                    [
                        HumanMessage(
                        content=[
                            {"type": "text", "text": image_analysis_prompt}] + imgs_content
                        )
                    ]
                )
        
        classification_result = result.Classification
        explanation_result = result.Explanation

        return {
            "id_pred": id_predict,
            "result": classification_result,
            "metadata": {
                "explanation": explanation_result
            }
        }
