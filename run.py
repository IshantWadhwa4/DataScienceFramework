import streamlit as st
import get_Data as gd
import EDA as eda
import preModeling as PM
import modeling
dataFrame = None
pipeline_dict = None
st.title('Data Science')

url = st.text_input("Enter Data Url(Must be csv file)")

if url:
    gd = gd.get_Data()
    dataFrame = gd.get_data_csv(url)
    pipeline_dict = gd.get_pipline_dict()
    eda = eda.EDA()
    pm = PM.preModeling()
    m = modeling.Modeling()
    #st.write(url)
    if st.checkbox('Basic EDA'):
        #st.text_input("Notes")
        eda.basic_EDA(dataFrame,pipeline_dict)

    if st.checkbox('Pre Modeling'):
        pm.do_pre_modeling(dataFrame,pipeline_dict)

    if st.checkbox('Modeling'):
        m.do_modeling(dataFrame,pipeline_dict)
