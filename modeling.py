import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from pycaret import classification as cl
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

X = None
y = None
finalize_model =None

class Modeling():

    def __init__(self):
        print('Modeling Start...')

    def do_modeling(self,dataFrame,pipeline_dict):

        prob_type = st.selectbox('Select type of problem' ,['Classification','Regression'])
        target_variable = st.selectbox('Select target variable',dataFrame.columns)

        classification_model_library = ['lr', 'knn', 'nb', 'dt', 'svm',
        'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda',
        'et', 'xgboost', 'lightgbm', 'catboost']

        tree_based_models = ['Random Forest Classifier',
                          'Decision Tree Classifier',
                          'Extra Trees Classifier',
                          'Gradient Boosting Classifier',
                          'Extreme Gradient Boosting',
                          'Light Gradient Boosting Machine',
                          'CatBoost Classifier']

        classification_model_names = ['Logistic Regression',
                       'K Neighbors Classifier',
                       'Naive Bayes',
                       'Decision Tree Classifier',
                       'SVM - Linear Kernel',
                       'SVM - Radial Kernel',
                       'Gaussian Process Classifier',
                       'MLP Classifier',
                       'Ridge Classifier',
                       'Random Forest Classifier',
                       'Quadratic Discriminant Analysis',
                       'Ada Boost Classifier',
                       'Gradient Boosting Classifier',
                       'Linear Discriminant Analysis',
                       'Extra Trees Classifier',
                       'Extreme Gradient Boosting',
                       'Light Gradient Boosting Machine',
                       'CatBoost Classifier']

        classification_models = dict(zip(classification_model_names,classification_model_library))

        if st.checkbox('X and y Split'):
            X = self.get_features(dataFrame,target_variable)
            y = dataFrame[target_variable]
            st.write('Done!')

        if st.checkbox('X,y Info'):
            st.write(X)
            st.write(y)

        if st.checkbox('Scaling of data'):
            scale_X = self.do_standardScale(X)
            columns = X.columns
            pipeline_dict['Scaling'] = True
            for col in scale_X:
                X[col] = scale_X[col].values
            #X.drop(columns,axis=1,inplace=True)
            st.write(X)
            st.write('Done!')


        if st.checkbox('Dimensionality Reduction'):
            if st.checkbox('PCA'):
                information_loss = st.text_input('Enter Information loss in percentage(%)')

                if st.button('PCA'):
                    pipeline_dict['PCA_info_loss'] = information_loss
                    pca_X = self.dimred_PCA(X,information_loss)
                    columns = X.columns
                    for i,val in enumerate(pca_X.T):
                        X[i] = val
                    X.drop(columns,axis=1,inplace=True)
                    st.write('Done!')

            if st.checkbox('LDA'):
                number_components = st.text_input('Enter the number of components')
                if st.button('LDA'):
                    pipeline_dict['LDA_number_components'] = number_components
                    lda = LDA( n_components = number_components)
                    lda_X = lda.fit_transform(X,y)
                    columns = X.columns
                    for i,val in enumerate(lda_X.T):
                        X[i] = val
                    X.drop(columns,axis=1,inplace=True)
                    st.write('Done!')



        if st.checkbox('Start Base-Line modeling Classification'):
            py_data = X
            py_data[target_variable] = y
            st.write('Name :' + str(target_variable))
            st.write('Type :' + str(prob_type))
            if st.checkbox('Start Modeling'):
                exp1 = cl.setup(data = py_data, target = target_variable, session_id=123, silent=True)
                st.write('Compare Models...')
                #models_info = cl.create_model('lr',verbose = False)
                models_info = cl.compare_models()
                st.write(models_info)
            if st.checkbox('Tuning Models'):
                tuning_model_name = st.selectbox('Select Model for Tuning',classification_model_names)
                if st.button('Start'):
                    st.write(tuning_model_name)
                    tuned_model,result = cl.tune_model(classification_models[tuning_model_name],verbose=False)
                    st.write(result)
                    if tuning_model_name in tree_based_models:
                        cl.interpret_model(tuned_model)
                        st.pyplot()
                        cl.plot_model(tuned_model, plot = 'confusion_matrix' )
                        st.pyplot()
                    else:
                        cl.plot_model(tuned_model, plot = 'confusion_matrix' )
                        st.pyplot()

            if st.checkbox('Finalize Model'):
                final_model_name = st.selectbox('Select Model for Tuning',classification_model_names)
                if st.checkbox('Finalize'):
                    tuned_model,result = cl.tune_model(classification_models[final_model_name],verbose=False)
                    st.write(result)
                    finalize_model = cl.finalize_model(tuned_model)
                    st.write(final_model_name)
                    st.write(finalize_model.get_params())
                    st.write('Done!')
                    st.write(pipeline_dict)
                    url = st.text_input("Enter Test Data Url(Must be csv file)")

                    if st.button('Click'):
                        test_dataFrame = self.get_test_data_csv(url)
                        st.write(test_dataFrame)
                        for k,v in pipeline_dict.items():
                            if k == 'Convert_Data_Type':
                                st.write('Convert_Data_Type')
                                self.convert_type(test_dataFrame, pipeline_dict['Convert_Data_Type']['column_name'],
                                pipeline_dict['Convert_Data_Type']['data_type'])

                            elif k == 'remove_columns':
                                st.write('remove_columns')
                                test_dataFrame.drop(pipeline_dict['remove_columns'],axis=1,inplace=True)

                            elif k == 'remove_columns_threshold':
                                st.write('remove_columns_threshold..')
                                for threshold in pipeline_dict['remove_columns_threshold']:
                                    remove_columns = self.remove_null_columns(test_dataFrame, float(threshold))
                                    test_dataFrame.drop(remove_columns,axis=1,inplace=True)

                            elif k == 'Fill_Median_Mode_Columns':
                                st.write('Fill_Median_Mode_Columns..')
                                test_dataFrame = self.replace_null_columns(test_dataFrame,pipeline_dict['Fill_Median_Mode_Columns'])

                            elif k == 'Create_Bins':
                                st.write('Create_Bins..')
                                column = pipeline_dict['Create_Bins']['column_Name']
                                bins = pipeline_dict['Create_Bins']['Numbers_bin']
                                for i,c in enumerate(column):
                                    test_dataFrame[c] = self.do_bining(test_dataFrame,c,int(bins[i]))

                            elif k == 'OneHotEncoding':
                                st.write('OneHotEncoding..')
                                list_columns = pipeline_dict['OneHotEncoding']
                                for col in list_columns:
                                    tempdf = pd.get_dummies(data = test_dataFrame[col])
                                    for in_col in tempdf.columns:
                                        colName = str(col) +'_'+str(in_col)
                                        test_dataFrame[colName] = tempdf[in_col].values
                                test_dataFrame.drop(list_columns,axis=1,inplace=True)

                            elif k == 'LabelEncoding':
                                st.write('LabelEncoding..')
                                test_dataFrame = self.do_label_Encoding(test_dataFrame,pipeline_dict['LabelEncoding'])

                            elif k == 'BinaryEncoding':
                                st.write('BinaryEncoding..')
                                binary_encoding_columns = pipeline_dict['BinaryEncoding']
                                for col in binary_encoding_columns:
                                    encoder = ce.BinaryEncoder(cols=[col])
                                    dfbin = encoder.fit_transform(dataFrame[col])
                                    for col in dfbin.columns:
                                        test_dataFrame[col] = dfbin[col].values
                                test_dataFrame.drop(binary_encoding_columns,axis=1,inplace=True)

                            elif k == 'Scaling':
                                st.write('Scaling..')
                                scale_X = self.do_standardScale(test_dataFrame)
                                columns = test_dataFrame.columns
                                for col in scale_X:
                                    test_dataFrame[col] = scale_X[col].values


                        st.write(test_dataFrame)
                        unseen_predictions = cl.predict_model(finalize_model, data=test_dataFrame)
                        st.write(unseen_predictions.head())
                        unseen_predictions.to_csv('result.csv')


    @st.cache(allow_output_mutation=True)
    def get_features(self,dataFrame,target_variable):
        return dataFrame.drop(target_variable,axis = 1)

    @st.cache(allow_output_mutation=True)
    def get_test_data_csv(self,url):
        #print(url)
        return pd.read_csv(filepath_or_buffer = url)

    def convert_type(self,df,list_column,list_type):
        for k,col in enumerate(list_column):
            df[col] = df[col].astype(list_type[k])
        return df

    def do_bining(self,df,column_name,number_bins,list_lables = False):
        return pd.cut(df[column_name],bins=number_bins,labels= list_lables)

    def do_label_Encoding(self,df,column_list):
        encode = LabelEncoder()
        for col in column_list:
            if isinstance(df[col].dtype, object):
                df[col] = encode.fit_transform(df[col])
        return df


    def get_catagorical_data_columns(self,df):
        return list(set(df.columns) - set(df._get_numeric_data().columns))

    def get_numeric_data_columns(self,df):
        return list(df._get_numeric_data().columns)

    #@st.cache(allow_output_mutation=True)
    def do_standardScale(self,X):
        sc = StandardScaler()
        sc.fit(X)
        x_sc = sc.transform(X)
        x_sc_df = pd.DataFrame(x_sc,columns=X.columns)
        return x_sc_df

    def replace_null_columns(self,df,list_columns):
        numeric_columns = self.get_numeric_data_columns(df)
        catagoric_columns = self.get_catagorical_data_columns(df)
        for col in list_columns:
            if col in numeric_columns:
                df[col].fillna(df[col].median(),inplace=True)
            elif col in catagoric_columns:
                 df[col].fillna(df[col].mode()[0],inplace=True)
        return df

    def remove_null_columns(self,df,threshold):
        null_values = self.get_number_zeros_null(df)
        null_values.loc['null_percantage'] = (null_values.loc['Number_of_nulls']/df.shape[0])* 100
        drop_column = []
        for col in null_values.columns:
            if null_values.loc['null_percantage',col] >= threshold:
                drop_column.append(col)
        return drop_column


    def dimred_PCA(self,X,information_loss):
        '''
        input: information_loss how much info loss is good for you in percentage, X is the dependent variables/columns dataframe
        '''
        info = 1 - (int(information_loss)/100)
        pca = PCA(info).fit(X)
        self.Variance_Explained_PCA_graph(pca)
        print('number of columns left are: {}'.format(pca.n_components_))
        transform_df = pca.transform(X)
        return transform_df


    def Variance_Explained_PCA_graph (self,pca):
        var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
        plt.ylabel('% Variance Explained')
        plt.xlabel('Number of Features')
        plt.title('PCA Analysis')
        plt.ylim(30,100.5)
        plt.style.context('seaborn-whitegrid')
        plt.plot(var)
        st.pyplot()
