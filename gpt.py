import os
from apikey import apikey
import streamlit as st
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from googletrans import Translator

# TamilGPT app
os.environ['OPENAI_API_KEY'] = apikey
st.title('TamilGPT: English to Tamil translation')
prompt = st.text_input('Ask a question in English for a Tamil translation and an answer. Ask anything!')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Translate this question into Tamil first: {topic}'
)

answer_template = PromptTemplate(
    input_variables=['answer'],
    template='Answer this question in English: {answer}'
)

translate_template = PromptTemplate(
    input_variables=['title', 'translate_research'],
    template='Translate the answer "{title}" to Tamil(while spelling it out phonetically) using Google Translate: {translate_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
answer_memory = ConversationBufferMemory(input_key='answer', memory_key='chat_history')
translate_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMS
llm = OpenAI(temperature=0.9)  # How creative the model can be
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
answer_chain = LLMChain(llm=llm, prompt=answer_template, verbose=True, output_key='answer', memory=answer_memory)
translate_chain = LLMChain(llm=llm, prompt=translate_template, verbose=True, output_key='translate_answer', memory=translate_memory)

# Google Translate
translator = Translator(service_urls=['translate.google.com'])

if prompt:
    title = title_chain.run({'topic': prompt})
    answer = answer_chain.run({'answer': prompt})
    translate_research = translator.translate(answer, dest='ta').text
    translate = translate_chain.run({'title': title, 'translate_research': translate_research})

    st.write('Question in Tamil:')
    st.write(title)
    st.write('Answer in English:')
    st.write(answer)
    st.write('Phonetic translation of Tamil in English(for us new learners):')
    st.write(translate)
    st.write('Answer in Tamil:')
    st.write(translate_research)


    with st.expander('Answer in TAMIL'):
        st.info(translate)
        

    with st.expander('Topic History'):
        st.info(title_memory.buffer)

    with st.expander('Answer History(English)'):
        st.info(answer_memory.buffer)

    with st.expander('Translate History'):
        st.info(translate_memory.buffer)

    
