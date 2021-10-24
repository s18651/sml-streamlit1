import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib as plt
import os
from transformers import pipeline

st.set_page_config(page_title='Streamlit by s18651', page_icon=None, layout='centered', initial_sidebar_state='expanded', menu_items=None)

st.title('Streamlit')

st.image("uk-deu-flags.jpg")
st.header("Aplikacja umożliwia:")
st.subheader('Tłumaczenie języka angielskiego na niemiecki')
st.subheader('Wydźwięk emocjonalny tekstu (eng)')

option = st.button("Wydźwięk emocjonalny tekstu (eng) oraz przetłumacz z angielskiego na niemiecki")

text = st.text_area(label="Wpisz tekst")

with st.spinner('In progress! ( ͡👁️ ͜ʖ ͡👁️)'):
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)

        classifier = pipeline("translation_en_to_de")
        answer = classifier(text)
        st.write(answer)


data = pd.DataFrame({
    'Capitals' : ['London', 'Berlin'],
    'lat' : [51.509865, 52.520008],
    'lon' : [-0.118092, 13.404954]
})

data
