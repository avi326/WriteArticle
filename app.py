from langchain.llms.openai import OpenAI
from langchain import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the OpenAI key from the environment
openai_api_key = os.getenv("OPENAI_KEY")

prompt_template = """
1. Analyzing the article writing style
2. generate the new article in hebrew.

please print just the new article.

"article":
{article}

"""

input_article = """ 
אובדן שליטה בחברה הערבית - וזעם ביישוב יפיע, שם התרחש הטבח שגבה את חייהם של 5 בני אדם. "לפני שהמשטרה חקרה, היא קישרה למשפחת פשע", תקף ב-ynet radio האני מרג'יה, דודם של שניים מהנרצחים. ראש המועצה לאולפן ynet: "בכל מדינה מתוקנת השר שאחראי על המשטרה היה מתפטר" 
 """

# Initialize the OpenAI module, load and run the summarize chain
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

prompt = PromptTemplate(
    input_variables=["article"],
    template=prompt_template,
)

final_prompt = prompt.format(article=input_article)

output = llm(final_prompt)

print(output)
