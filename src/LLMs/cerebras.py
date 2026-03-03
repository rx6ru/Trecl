from langchain_cerebras import ChatCerebras
from pydantic import SecretStr

from configs.llms import CEREBRAS_API_KEY

CEREBRAS = ChatCerebras(
    model="gpt-oss-120b",
    api_key=SecretStr(CEREBRAS_API_KEY),
)
