from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

# response = model.invoke("Tell me something about sun in 1 line")
# print(response.content)
