import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from pycaret import classification as cl

X = None
y = None

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
                    pca_X = self.dimred_PCA(X,information_loss)
                    columns = X.columns
                    for i,val in enumerate(pca_X.T):
                        X[i] = val
                    X.drop(columns,axis=1,inplace=True)
                    st.write('Done!')

            if st.checkbox('LDA'):
                number_components = st.text_input('Enter the number of components')
                if st.button('LDA'):
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
                    st.write(classification_models[tuning_model_name])
                    tuned_model,result = cl.tune_model(classification_models[tuning_model_name],verbose=False)
                    st.write(result)
                    cl.evaluate_model(tuned_model)
                    st.pyplot()
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
                if st.button('Finalize'):
                    tuned_model,result = cl.tune_model(classification_models[final_model_name],verbose=False)
                    st.write(result)
                    finalize_model = cl.finalize_model(tuned_model)
                    st.write(final_model_name)
                    st.write(finalize_model.get_params())
                    st.write('Done!')
                    st.write(pipeline_dict)





    @st.cache(allow_output_mutation=True)
    def get_features(self,dataFrame,target_variable):
        return dataFrame.drop(target_variable,axis = 1)

    #@st.cache(allow_output_mutation=True)
    def do_standardScale(self,X):
        sc = StandardScaler()
        sc.fit(X)
        x_sc = sc.transform(X)
        x_sc_df = pd.DataFrame(x_sc,columns=X.columns)
        return x_sc_df


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
