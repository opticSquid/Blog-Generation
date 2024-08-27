import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLAMA 2 model


def getLLamaResponse(input_text, no_words, blog_style):
    # LLama 2 model
    llm = CTransformers(
        model='models/llama-2-7b-chat.Q8_0.gguf', model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.01})

    # Prompt template
    template = """
         write a blog for {style} job profile for a topic {text} within {n_words}
         """
    prompt = PromptTemplate(
        input_variables=["style", "text", "n_words"], template=template)

    # Generate response from Llama GGUF model
    response = llm(prompt.format(style=blog_style,
                   text=input_text, n_words=no_words))
    print(response)
    return response


st.set_page_config(page_title="Generate Blogs", page_icon='🟆✦',
                   layout="centered", initial_sidebar_state="collapsed")
st.header("Generate Blogs 🟆✦")
input_text = st.text_input("Enter blog topic")
# creating 2 more columns for getting additional information
col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input("Number of words: ")
with col2:
    blog_style = st.selectbox(
        "Writing the blog for", ('Researchers', 'DataScientist', 'Common People'), index=0)
submit = st.button("Generate")
if submit:
    st.write(getLLamaResponse(input_text, no_words, blog_style))
