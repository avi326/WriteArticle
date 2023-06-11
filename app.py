import streamlit as st
from langchain.llms.openai import OpenAI
from langchain import PromptTemplate
from dotenv import dotenv_values

config = dotenv_values(".env")
openai_api_key = config["OPENAI_KEY"]

template = """
1. Analyzing the article writing style
2. rewrite it in hebrew. use "new information" to generate the new one.

"article":
{article}

"new information":
{info}
"""

# Streamlit app
st.title('LangChain Text Summarizer')

source_text = st.text_area("Source Text", height=200)
keywords_text = st.text_area("keywords data", height=200)

# Check if the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not source_text.strip() or not keywords_text:
        st.error(f"Please provide the missing fields.")
    else:
        try:

            # Initialize the OpenAI module, load and run the summarize chain
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

            prompt = PromptTemplate(
                input_variables=["article", "info"],
                template=template,
            )

            final_prompt = prompt.format(article=source_text, info=keywords_text)

            # Display summary
            st.success(llm(final_prompt))
        except Exception as e:
            st.error(f"An error occurred: {e}")
