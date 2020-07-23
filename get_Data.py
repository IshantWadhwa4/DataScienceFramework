import pandas as pd
import streamlit as st

class get_Data():
    def __init__(self):
        print('Start getting data....')

    @st.cache(allow_output_mutation=True)
    def get_data_csv(self,url):
        #print(url)
        return pd.read_csv(filepath_or_buffer = url)

    @st.cache(allow_output_mutation=True)
    def get_pipline_dict(self):
        return {}
