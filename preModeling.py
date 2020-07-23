import pandas as pd
import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
import math
import io
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce



class preModeling():

    def __init__(self):
        print('In PreModeling....')


    def do_pre_modeling(self,dataFrame,pipeline_dict):

        if st.checkbox('Solve for Null values'):
            if st.checkbox('Remove Columns if Null greater than threshold(%)'):
                threshold = st.text_input('Enter threshold Value in percentage(%)')
                if st.button('Remove'):
                    if 'remove_columns_threshold' in pipeline_dict.keys():
                        pipeline_dict['remove_columns_threshold'].append(threshold)
                    else:
                        pipeline_dict['remove_columns_threshold'] = [threshold]
                    remove_columns = self.remove_null_columns(dataFrame, float(threshold))
                    dataFrame.drop(remove_columns,axis=1,inplace=True)
                    st.write('List of droped columns')
                    st.write(remove_columns)

            if st.checkbox('Fill Null value with median and mode'):
                columns = st.multiselect('Select Columns',dataFrame.columns)
                if st.button('Fill'):
                    if 'Fill_Median_Mode_Columns' in pipeline_dict.keys():
                        for c in columns:
                            pipeline_dict['Fill_Median_Mode_Columns'].append(c)
                    else:
                        pipeline_dict['Fill_Median_Mode_Columns'] = columns
                    dataFrame = self.replace_null_columns(dataFrame,columns)
                    st.write('Done')

            if st.checkbox('Using Modeling'):
                st.write('ToDo')

        if st.checkbox('Solve for Outliars'):
            column = st.selectbox('Select Column',self.get_numeric_data_columns(dataFrame))
            if st.checkbox('Show boxplot'):
                self.draw_single_boxplot(dataFrame,column)
            if st.checkbox('Click If Outliars'):
                rows = st.text_input('See top n values for above Selected Column')
                if st.checkbox('Show n largest values'):
                    st.write(dataFrame[column].nlargest(int(rows)))
                if st.checkbox('Want to remove Rows'):
                    value = st.text_input('Enter Outliar Value')
                    if st.button('Remove Rows greater than above value'):
                        index_list = dataFrame[dataFrame[column] >= int(value)].index
                        dataFrame.drop(index_list,axis=0,inplace = True)
                        st.write('Done!')

        if st.checkbox('Convert continous data into classes by bins'):
            column = st.selectbox('Select Column',self.get_numeric_data_columns(dataFrame))
            bins = st.text_input('Enter Number of bins')

            if st.button('Convert'):
                if 'Create_Bins' in pipeline_dict.keys():
                    pipeline_dict['Create_Bins']['column_Name'].append(column)
                    pipeline_dict['Create_Bins']['Numbers_bin'].append(column)
                else :
                    pipeline_dict['Create_Bins'] = {'column_Name':[column],'Numbers_bin':[bins]}
                dataFrame[column] = self.do_bining(dataFrame,column,int(bins))
                st.write('Done!')

        #if st.checkbox('Important Features Selection'):
        #    target_column = st.selectbox('Select the target column(Classification)',self.df.columns)
        #    max_features = st.text_input('Enter max features you want to select')
        #    if st.button('Important features'):
        #        st.write('List of important features are ')
        #        st.write(self.randomForestBased_FeatureImportance,self.df,target_column,int(max_features))

        if st.checkbox('Encoding'):
            if st.checkbox('One-hot Encoding'):
                list_columns = st.multiselect('Columns for Encoding',dataFrame.columns)

                if st.button('Click OneHot'):
                    if 'OneHotEncoding' in pipeline_dict.keys():
                        for c in list_columns:
                            pipeline_dict['OneHotEncoding'].append(c)
                    else:
                        pipeline_dict['OneHotEncoding'] = list_columns
                    for col in list_columns:
                        tempdf = pd.get_dummies(data = dataFrame[col])
                        for in_col in tempdf.columns:
                            colName = str(col) +'_'+str(in_col)
                            dataFrame[colName] = tempdf[in_col].values
                    dataFrame.drop(list_columns,axis=1,inplace=True)
                    st.write('Done!')

            if st.checkbox('Label Encoding'):
                label_encoding_columns = st.multiselect('Columns for Encoding',dataFrame.columns)

                if st.button('Label Encoding'):
                    if 'LabelEncoding' in pipeline_dict.keys():
                        for c in label_encoding_columns:
                            pipeline_dict['LabelEncoding'].append(c)
                    else:
                        pipeline_dict['LabelEncoding'] = label_encoding_columns
                    dataFrame = self.do_label_Encoding(dataFrame,label_encoding_columns)
                    st.write('Done!')


            if st.checkbox('Binary encoding'):
                binary_encoding_columns = st.multiselect('Columns for Encoding',dataFrame.columns)

                if st.button('Binary Encoding'):
                    if 'BinaryEncoding' in pipeline_dict.keys():
                        for c in binary_encoding_columns:
                            pipeline_dict['BinaryEncoding'].append(c)
                    else:
                        pipeline_dict['BinaryEncoding'] = binary_encoding_columns
                    for col in binary_encoding_columns:
                        encoder = ce.BinaryEncoder(cols=[col])
                        dfbin = encoder.fit_transform(dataFrame[col])
                        for col in dfbin.columns:
                            dataFrame[col] = dfbin[col].values
                    dataFrame.drop(binary_encoding_columns,axis=1,inplace=True)
                    st.write('Done!')




    def do_label_Encoding(self,df,column_list):
        encode = LabelEncoder()
        for col in column_list:
            if isinstance(df[col].dtype, object):
                df[col] = encode.fit_transform(df[col])
        return df


    def randomForestBased_FeatureImportance(self,df,target_column,max_features):
        X = df.drop(target_column,axis = 1)
        y= df[target_column]
        embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=max_features)
        embeded_rf_selector.fit(X, y)
        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
        return embeded_rf_feature

    def do_bining(self,df,column_name,number_bins,list_lables = False):
        return pd.cut(df[column_name],bins=number_bins,labels= list_lables)



    def draw_single_boxplot(self,df,column):
        #fig=plt.figure(num=None, figsize=(8, 10), dpi=80, facecolor='w', edgecolor='k')
        sns.boxplot( y=df[column],palette="Set3")
        #fig.tight_layout()  # Improves appearance a bit.
        st.pyplot()



    def replace_null_columns(self,df,list_columns):
        numeric_columns = self.get_numeric_data_columns(df)
        catagoric_columns = self.get_catagorical_data_columns(df)
        for col in list_columns:
            if col in numeric_columns:
                df[col].fillna(df[col].median(),inplace=True)
            elif col in catagoric_columns:
                 df[col].fillna(df[col].mode()[0],inplace=True)
        return df

    def get_catagorical_data_columns(self,df):
        return list(set(df.columns) - set(df._get_numeric_data().columns))

    def get_numeric_data_columns(self,df):
        return list(df._get_numeric_data().columns)

    def remove_null_columns(self,df,threshold):
        null_values = self.get_number_zeros_null(df)
        null_values.loc['null_percantage'] = (null_values.loc['Number_of_nulls']/df.shape[0])* 100
        drop_column = []
        for col in null_values.columns:
            if null_values.loc['null_percantage',col] >= threshold:
                drop_column.append(col)
        return drop_column

    def get_number_zeros_null(self,df):
        null_zero_dict={ }
        null_zero_dict['Number_of_nulls'] = df.isnull().sum()
        null_zero_dict['Number_of_zeros'] = (df==0).astype(int).sum()
        return pd.DataFrame(null_zero_dict).T
