import pandas as pd
import numpy as np
def class_imbalance(df: pd.DataFrame, feature_name: str, feature_value: str) -> float:
    n_d = len(df[df[feature_name] == feature_value])
    n_a = len(df[df[feature_name] != feature_value])
    ci = (n_a - n_d)/(n_a + n_d)
    return ci


def dpl(df: pd.DataFrame, feature_name: str, feature_value: str, label_name: str, label_value:str) -> float:
    n_d = len(df[df[feature_name] == feature_value])
    n_a = len(df[df[feature_name] != feature_value])

    # p(label_value|not feature_value) or q_a
    p_positive_given_a = len(
        df[(df[feature_name] != feature_value) & (df[label_name] == label_value)]
    ) / n_a
    # p(label_value|feature_value) or q_d
    p_positive_given_d = len(
        df[(df[feature_name] == feature_value) & (df[label_name] == label_value)]
    ) / n_d
    return p_positive_given_a - p_positive_given_d


def tvd(
    df: pd.DataFrame,
    feature_name: str,
    feature_value: str,
    label_name: str,
    positive_label_value: str,
    negative_label_value: str
) -> float:
    n_d = len(df[df[feature_name] == feature_value])
    n_a = len(df[df[feature_name] != feature_value])

    p_positive_given_d = len(
        df[(df[feature_name] == feature_value) & (df[label_name] == positive_label_value)]
    ) / n_d
    p_negative_given_d = len(
        df[(df[feature_name] == feature_value) & (df[label_name] == negative_label_value)]
    ) / n_d

    p_positive_given_a = len(
        df[(df[feature_name] != feature_value) & (df[label_name] == positive_label_value)]
    ) / n_a
    p_negative_given_a = len(
        df[(df[feature_name] != feature_value) & (df[label_name] == negative_label_value)]
    ) / n_a


    tvd_value = 0.5 * ((np.abs(p_positive_given_d - p_positive_given_a)
                               + np.abs(p_negative_given_d - p_negative_given_a)))
    return tvd_value


def js_divergence(
    df: pd.DataFrame,
    feature_name: str,
    feature_value: str,
    label_name: str,
    positive_label_value: str,
    negative_label_value: str
) -> float:
    n_d = len(df[df[feature_name] == feature_value])
    n_a = len(df[df[feature_name] != feature_value])

    p_positive_given_d = len(
        df[(df[feature_name] == feature_value) & (df[label_name] == positive_label_value)]
    ) / n_d
    p_negative_given_d = len(
        df[(df[feature_name] == feature_value) & (df[label_name] == negative_label_value)]
    ) / n_d

    p_positive_given_a = len(
        df[(df[feature_name] != feature_value) & (df[label_name] == positive_label_value)]
    ) / n_a
    p_negative_given_a = len(
        df[(df[feature_name] != feature_value) & (df[label_name] == negative_label_value)]
    ) / n_a

    p_positive = 0.5 * (p_positive_given_a + p_positive_given_d)
    p_negative = 0.5 * (p_negative_given_a + p_negative_given_d)

    term1 = p_positive_given_a * np.log(p_positive_given_a/p_positive) + p_negative_given_a * np.log(p_negative_given_a/p_negative)
    term2 = p_positive_given_d * np.log(p_positive_given_d/p_positive) + p_negative_given_d * np.log(p_negative_given_d/p_negative)
    return 0.5 * (term1 + term2)




from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostClassifier
import dalex as dx
from aif360.sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

feature_descriptions = {

    "Gender": "Gender of passenger",
    "Age": "Age of the passenger",
    "HYpertension": "whether patient is suffering from hypertension or not",
    "ever_married" : "Martial status of patient",
    "Heart_disease":"whether patient is suffering from heart disease or not",
    "smoking_status":"smoking status of patient"
}

ele = -1
def sunil(x, dic):
    global ele
    if x in dic.keys():
        return dic[x]
    else:
        ele += 1
        dic[x] = ele
        return dic[x]
def somil(df):
    mapped = dict()
    for column in df.columns:
        if df[column].dtype == 'O':
            global ele
            ele = -1
            dic = dict()
            df[column] = df[column].apply(lambda x: sunil(x, dic))
            dic = {v:k for k, v in dic.items()}
            mapped[column] = dic
    return mapped


def train_model(df):
    somil(df)
    X=df.drop(['target','patientmasterkey'],axis=1)
    y=df[['target']]
    
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    clf =DecisionTreeClassifier()

    clf.fit(x_train,y_train)
    return clf

    



def explain_dash(df):
    model=train_model(df)
    #somil(df)
    X=df.drop(['target','patientmasterkey'],axis=1)
    y=df[['target']]
    #y=np.where(y=='yes',1,0)
   
    explainer = ClassifierExplainer(model,X, y, 

                                target = "target"
                                )
    db = ExplainerDashboard(explainer, 
                            title="VExplainer", # defaults to "Model Explainer"
                            #shap_interaction=True, # you can switch off tabs with bools
                            )
    from explainerdashboard import InlineExplainer
    return db




def spd(df, column=None, y_pred=None,minority=[]):
    y_pred = np.array(y_pred)
    def check(x):
        if x in minority:
            return 0
        return 1
    sdf = pd.DataFrame(data = y_pred, columns=['prediction'])
    sdf['ref'] = df[column].apply(lambda x: check(x))
    return (sdf[sdf['ref'] == 0]['prediction'].value_counts()[1]/sdf[sdf['ref'] == 0].shape[0]) - (sdf[sdf['ref'] == 1]['prediction'].value_counts()[1]/sdf[sdf['ref'] == 1].shape[0])




def di(df, column=None, y_pred=None,minority=[]):
    y_pred = np.array(y_pred)
    def check(x):
        if x in minority:
            return 0
        return 1
    sdf = pd.DataFrame(data = y_pred, columns=['prediction'])
    sdf['ref'] = df[column].apply(lambda x: check(x))
    return (sdf[sdf['ref'] == 0]['prediction'].value_counts()[1]/sdf[sdf['ref'] == 0].shape[0])/(sdf[sdf['ref'] == 1]['prediction'].value_counts()[1]/sdf[sdf['ref'] == 1].shape[0])





def shap_v(df,col_choice):
    import shap
    model=train_model(df)
    somil(df)
    X=df.drop(['target','patientmasterkey'],axis=1)
    y=df[['target']]
    
    
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_values.values = np.mean(shap_values.values, axis=0)
    #shap_values.data = pd.DataFrame(columns=data.keys(), data=shap_values.data)
    shap.plots.force(shap_values[0])
    shap.waterfall_plot(explainer.expected_value[0], shap_values[0], X.iloc[1])

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
# Add a title and intro text
st.title('DATA BIASING EXPLORER')
st.text('This is a web app to allow exploration of Healthcare Data')
# Create file uploader object
upload_file = st.file_uploader('Upload a file containing data regarding biasing')
# Check to see if a file has been uploaded
if upload_file is not None:
    # If it has then do the following:
    # Read the file to a dataframe using pandas
    df = pd.read_csv(upload_file)
    df.dropna(inplace=True)
    # Create a section for the dataframe statistics
    st.header('Statistics of Dataframe')
    st.write(df.drop(['patientmasterkey'],axis=1).describe())
    # Create a section for the dataframe header
    st.header('Sample of Dataframe')
    st.write(df.head())
    ls=df.columns.tolist()
    ls.insert(0,'ALL')
    st.header('Bias in dataset')
    col_choice = st.selectbox('Choose feature name:',ls)
    col_val=st.selectbox('Choose threshold:',df[col_choice].unique())
    df['target']=np.where(df['target']==1,'yes','no')
    st.write(px.histogram(df, x ='target', color = col_choice))
    st.subheader('class imbalance is:')
    st.write()
    st.write('Class imbalance (CI) bias occurs when a disadvantageous group has fewer training samples when compared with advantageous group in the dataset. ')
    st.write('Range=(-1,+1)')
    st.write()
    st.write('Positive CI values indicate the advantageous group has more training samples in the dataset and a value of 1 indicates the data only contains members of the advantageous group.')
    st.write()
    st.write('Values of CI near zero indicate a more equal distribution of members between facets and a value of zero indicates a perfectly equal partition between facets and represents a balanced distribution of samples in the training data.')
    st.write()
    st.write('Negative CI values indicate the disadvantageous group has more training samples in the dataset and a value of -1 indicates the data only contains members of the disadvantageous group.')
    st.write()
    st.write('CI values near either of the extremes values of -1 or 1 are very imbalanced and are at a substantial risk of making biased predictions.')
    st.write()
    st.metric(label="Class imbalance", value=round(class_imbalance(df,col_choice,col_val),3))
    #st.write(class_imbalance(df,col_choice,col_val))
    
    st.write()
    label_val=st.selectbox('choose target val:',df['target'].unique())
    st.subheader('Difference in positive proportion(DPL) is:')
    st.write()
    st.write('The difference in proportions of labels (DPL) compares the proportion of observed outcomes with positive labels for disadvantageous group with the proportion of observed outcomes with positive labels of advantageous group in a training dataset. ')
    st.write('Range=(-1,1)')
    st.write('')
    st.write('Positive DPL values indicate that advantageous group  has a higher proportion of positive outcomes when compared with disadvantageous group.')
    st.write('')
    st.write('Values of DPL near zero indicate a more equal proportion of positive outcomes between both group and a value of zero indicates perfect demographic parity.')
    st.write('')

    st.write('Negative DPL values indicate that disadvantageous group has a higher proportion of positive outcomes when compared with advantageous group.')
    st.write('')
    st.metric(label="DPL", value=round(dpl(df,col_choice,col_val,'target',label_val),3))
    #st.write(dpl(df,col_choice,col_val,'stroke',label_val))
  


    st.subheader('Jensen-Shannon Divergence (JS) is:')
    st.write()
    st.write('it measures how much the label distribution of different classes diverge from each other')
    st.write()
    st.write('The range of JS values for binary, multicategory, continuous outcomes is [0, ln(2)).')
    st.write()
    st.write('Values near zero mean the labels are similarly distributed.')
    st.write()
    st.write('Positive values mean the label distributions diverge, the more positive the larger the divergence.')
    st.write()
    st.write('This metric indicates whether there is a big divergence in one of the labels across facets.')
    st.write()
    st.metric(label="JS_divergence", value=round(js_divergence(df,col_choice,col_val, 'target', 'yes','no'),3))
    #st.write(js_divergence(df,col_choice,col_val, 'stroke', 1,0))
    
#     st.subheader('Total variation distance (TVD) is :')
#     st.write()
#     st.write(' The TVD is the largest possible difference between the probability distributions for label outcomes of advantageous and disadvantageous group')
#     st.write()
#     st.write('The range of TVD values for binary, multicategory, and continuous outcomes is [0, 1]')
#     st.write()
#     st.write('Values near zero mean the labels are similarly distributed.')
#     st.write()
#     st.write('Positive values mean the label distributions diverge, the more positive the larger the divergence.')
#     st.write()
#     st.metric(label="TVD", value=round(tvd(df,col_choice,col_val,'target','yes','no'),3))
    #st.write(tvd(df,col_choice,col_val,'stroke',1,0))
    
    st.subheader('Statistical parity difference is')
    st.write()
    st.write('Statistical Parity Difference (SPD)  metric is defined as the difference of the proportion of positive predictions (y’ = 1) for disadvantageous group over the proportion of positive predictions (y’ = 1) for advantageous group..')
    st.write('Under 0: Higher benefit for the monitored group.')
    st.write()
    st.write('At 0: Both groups have equal benefit.')
    st.write()
    st.write('Over 0 Implies higher benefit for the reference group.')
    st.write()
    st.metric(label="SPD", value=round(spd(df,column=col_choice,y_pred=df['target'],minority=[col_val]),3))
    #st.write(spd(df,column=col_choice,y_pred=df['stroke'],minority=[col_val]))
    st.write()
   
    st.write()
    st.subheader('Disperate impact is')
    st.write()
    st.write('The disparate impact (DI) metric is defined as the ratio of the proportion of positive predictions (y’ = 1) for disadvantageous group over the proportion of positive predictions (y’ = 1) for advantageous group.')
    st.write()
    st.write('For binary, multicategory facet, and continuous labels, the DI values range over the interval [0, ∞)')

    st.write('Values less than 1 indicate that facet a has a higher proportion of predicted positive outcomes than facet d. This is referred to as positive bias.')
    st.write()

    st.write('A value of 1 indicates demographic parity.')
    st.write()

    st.write('Values greater than 1 indicate that facet d has a higher proportion of predicted positive outcomes than facet a. This is referred to as negative bias.')
    st.write()
    st.metric(label="DI", value=round(di(df, column=col_choice,y_pred=df['target'], minority=[col_val]),3))
    #st.write(di(df, column=col_choice,y_pred=df['target'], minority=[col_val]))
    
#     st.subheader('Trying to run shap')
#     st.write(shap_v(df,col_choice))
#     st.subheader('Model fairness with respect to module Dalex')
#     dalex_bias(df)
    check_explainer=st.selectbox('Want to run explainer',['No','Yes'])
    if check_explainer=='Yes':
        
        db = explain_dash(df)
        url="http://127.0.0.1:8050/"
        st.write("check out this [link](%s)" % url)

        #st.markdown("check out this [link](%s)" % url)


        db.run(port=8050)
    
 