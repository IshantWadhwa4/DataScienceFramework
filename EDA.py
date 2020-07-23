import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
import math
import io

class EDA():

    def __init__(self):
        print('Starting EDA')

    def basic_EDA(self,dataFrame,pipeline_dict):
        if st.checkbox('Top 10 Rows'):
            st.write(dataFrame.head(10))

        if st.checkbox('Shape'):
            st.write("Number of Rows are "+ str(dataFrame.shape[0]))
            st.write("Number of Columns are "+str(dataFrame.shape[1]))

        if st.checkbox('Data info'):
            # todo result is none
            st.write(self.get_number_zeros_null(dataFrame))

        if st.checkbox('Data describe'):
            st.write(dataFrame.describe())

        if st.checkbox('Convert Data Type'):
            data_options = st.multiselect('Select Columns You want to change data type: ',dataFrame.columns)
            datatype = st.multiselect('Select datatype You want respectively: ',['object','int64','float64',
                                    'datetime64','bool'])

            if st.button('Change'):
                if 'Convert_Data_Type' in pipeline_dict.keys():
                    pipeline_dict['Convert_Data_Type']['column_name'].append(data_options)
                    pipeline_dict['Convert_Data_Type']['data_type'].append(datatype)
                else:
                    pipeline_dict['Convert_Data_Type'] = {'column_name':data_options,'data_type':datatype}
                print(data_options)
                self.convert_type(dataFrame,data_options,datatype)
                st.write('Done!')

        if st.checkbox('Remove columns'):
            options = st.multiselect('Select Columns You want to remove: ',dataFrame.columns)

            if st.button('Drop') :
                if 'remove_columns' in pipeline_dict.keys():
                    for o in options:
                        pipeline_dict['remove_columns'].append(o)
                else:
                    pipeline_dict['remove_columns'] = options
                dataFrame.drop(options,axis=1,inplace=True)
                st.write('Droped columns are')
                st.write(options)

        if st.checkbox('Draw catagorical Columns'):
            self.draw_countPlot_grid(dataFrame)

        if st.checkbox('Draw Numeric Columns'):
            self.draw_distributionPlot_grid(dataFrame)

        if st.checkbox('Co-relation'):
            if st.checkbox('Co-relation b/w all Columns'):
                self.heatmap_allcolumns(dataFrame)
            if st.checkbox('Co-relation b/w Columns with threshold'):
                pthreshold = st.text_input('Enter Pos-threshold value b/w 0 to 1 Example=0.90')
                nthreshold = st.text_input('Enter Neg-threshold value b/w 0 to 1 Example=0.90')
                if st.button('Show'):
                    self.create_seaborn_heatmap_highcorelated(dataFrame,float(pthreshold),float(nthreshold))

        if st.checkbox('Box-Plot'):
            self.draw_boxPlot_grid(dataFrame)

    def get_number_zeros_null(self,df):
        null_zero_dict={}
        null_zero_dict['DataType'] = df.dtypes
        null_zero_dict['Number_of_nulls'] = df.isnull().sum()
        null_zero_dict['Number_of_zeros'] = (df==0).astype(int).sum()
        return pd.DataFrame(null_zero_dict)

    def convert_type(self,df,list_column,list_type):
        for k,col in enumerate(list_column):
            df[col] = df[col].astype(list_type[k])
        return df

    def get_catagorical_data_columns(self,df):
        return list(set(df.columns) - set(df._get_numeric_data().columns))

    def draw_countPlot_grid(self,df):
        fig=plt.figure(num=None, figsize=(12, 15), dpi=80, facecolor='w', edgecolor='k')
        list_columns = self.get_catagorical_data_columns(df)
        n_rows = math.ceil(len(list_columns)/3)
        n_cols = 3
        count = 0
        for  var_name in list_columns:
            if len(df[var_name].unique()) < 8:
                ax=fig.add_subplot(n_rows,n_cols,count + 1)
                count += 1
                sns.countplot(x = var_name, data=df)
                ax.set_title(var_name)
        fig.tight_layout()  # Improves appearance a bit.
        st.pyplot()

    def convert_type(self,df,list_column,list_type):
        print(list_column)
        for k,col in enumerate(list_column):
            print(k,col)
            if list_type[k] == 'datetime64':
                df[col] = pd.to_datetime(df[col])
            else:
                df[col] = df[col].astype(list_type[k])
        return df

    def get_numeric_data_columns(self,df):
        return list(df._get_numeric_data().columns)

    def draw_distributionPlot_grid(self,df):
        fig=plt.figure(num=None, figsize=(12,15), dpi=80, facecolor='w', edgecolor='k')
        list_columns = self.get_numeric_data_columns(df)
        n_rows = math.ceil(len(list_columns)/3)
        n_cols = 3
        colors = []

        for i in range(n_rows*n_cols):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

        for i, var_name in enumerate(list_columns):
            ax=fig.add_subplot(n_rows,n_cols,i+1)
            sns.distplot(df[var_name],hist=True,axlabel=var_name,color=colors[i],kde_kws={'bw':1.5})
            ax.set_title(var_name)

        fig.tight_layout()  # Improves appearance a bit.
        st.pyplot()

    def heatmap_allcolumns(self,df):
        fig=plt.figure(num=None, figsize=(25, 20), dpi=80, facecolor='w', edgecolor='k')
        sns.heatmap(data=df.corr(),annot=True, cmap="Blues")
        st.pyplot()

    def create_seaborn_heatmap_highcorelated(self,df,posThreshold,negThreshold):
        #fig=plt.figure(num=None, figsize=(25, 20), dpi=80, facecolor='w', edgecolor='k')
        df_corr = df.corr()
        tempdf = df_corr[(df_corr > posThreshold) | (df_corr < -negThreshold)]
        tempdf.replace(to_replace=1,value=np.nan,inplace=True)
        tempdf.dropna(axis=1,how='all',inplace=True)
        tempdf.dropna(axis=0,how='all',inplace=True)
        #st.write(tempdf)
        if tempdf.empty:
            st.write('No feature, above or equal to this threshold. Please change the value. ')
        else:
            sns.heatmap(tempdf,annot=True, cmap="Blues")
            st.pyplot()

    def draw_boxPlot_grid(self,df):
        fig=plt.figure(num=None, figsize=(12, 15), dpi=80, facecolor='w', edgecolor='k')
        list_columns = self.get_numeric_data_columns(df)
        n_rows = math.ceil(len(list_columns)/3)
        n_cols = 3
        for i, var_name in enumerate(list_columns):
            ax=fig.add_subplot(n_rows,n_cols,i+1)
            sns.boxplot( y=df[var_name])
            ax.set_title(var_name)
        fig.tight_layout()  # Improves appearance a bit.
        st.pyplot()
