from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the OpenAI key from the environment
openai_api_key = os.getenv("OPENAI_KEY")

prompt_template_for_generate_new_article = """
You are a helpful assistant that expert in hebrew language. 
please do this things when the user write a article in hebrew: 

1. Analyzing the article writing style
2. generate the new article in hebrew.

please print just the generated article.
"""

prompt_template_for_analysis = """
You are a helpful assistant that expert in hebrew language. 
please do this things when the user write a article in hebrew: 

1. count number of words
2. print the entities and thier types
3. print sentiment and say why you think like this.

"""


input_article = """ 
אובדן שליטה בחברה הערבית - וזעם ביישוב יפיע, שם התרחש הטבח שגבה את חייהם של 5 בני אדם. "לפני שהמשטרה חקרה, היא קישרה למשפחת פשע", תקף ב-ynet radio האני מרג'יה, דודם של שניים מהנרצחים. ראש המועצה לאולפן ynet: "בכל מדינה מתוקנת השר שאחראי על המשטרה היה מתפטר" 
 """

# Initialize the OpenAI module
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
messages = [
    SystemMessage(content=prompt_template_for_analysis),
    HumanMessage(content=input_article)
]

response = llm(messages)

print(response.content, end='\n')
