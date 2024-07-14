###############Importing###############
#####Import the required libraries for the machine learning application.
from numpy.core.numeric import True_
import streamlit as st
import sklearn
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
import shap
import matplotlib
import matplotlib.pyplot as pl
import altair as alt
import seaborn as sns
import pickle
matplotlib.use('Agg')

###############Initialization###############
#Let’s start very simple by creating a python script that prints text to the Web application.
def main():
    st.title("Explaining the Random Forest models-based customer satisfaction predictions")
    st.write("Made By: Zibin Zhao")
    st.header("Explanatory information - Random Forest Models")
    st.image("https://raw.githubusercontent.com/zibinzhao/Interactivity/main/Picture_first_page.png")
              
    st.sidebar.title("Create your model")
if __name__ == '__main__':
    main()
    
###############Data Loading###############
@st.cache(persist=True)
def load():
    data= pd.read_csv("https://raw.githubusercontent.com/zibinzhao/Interactivity/main/df_i.csv")
    label= LabelEncoder()
    for i in data.columns:
        data[i] = label.fit_transform(data[i])
        return data
df = load()

st.sidebar.subheader("Display the Fast_Delivery Dataset")
if st.sidebar.checkbox("Display data", False):
    st.subheader("Show dataset")
    st.write(df)
    
    ###############Data Analysis###############
    ####Count of the Key variable
    variable_selector = st.sidebar.selectbox('Key variable:',df.columns, index=1)
    st.subheader("Count of the Key variable")
    with st.expander("Additional descriptions about variables"):
        st.write("1) **level**: The customer satisfaction levels which grouped by the value of customer satisfaction scores: 0 - Unsatisfied (1,2,3,4,5); 1 - Satisfied (6,7,8,9).")
        st.write("2) **sat**: Latest overall satisfaction score of the customer: 0 – 10, where 10 is extremely satisfied.")
        st.write("3) **ordered**: Number of times the customer order food on (or use) this platform")
        st.write("4) **age**: Age of the customer")
        st.write("5) **speed**: The delivery speed that the customer received the food: 1 - slow; 2 - normal; 3 - fast")
        st.write("6) **single**: 1 - “yes” if the customer is single; 2 - “no” if the customer is not single")
        st.write("7) **dist**: Distance of the customer’s home to the nearest city centre")
        st.write("8) **income**: Income of the respective customer in £")
        st.write("9) **range**: The income ranges which classified by the value of cusomter's income: 1 - Low income (0<income<=80000); 2 - Medium income (80000<income<=120000); 3 - High income (120000<income<=140000)")
    st.write('Count of variable:',Counter(df[variable_selector]))
    sns.countplot(x=variable_selector, data=df)
    pl.title('The count of the key varaible', fontsize=10)
    st.pyplot()

    ####Visualizing correlations - Cutomer satisfaction Correlation Heatmap
    st.subheader("Visualising correlations")
    st.write("Visualising the correlations between each variable by heatmap")
    f, ax = pl.subplots(figsize=(10, 6))
    corr = df.corr()
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="Reds",fmt='.2f',
            linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle('Cutomer satisfaction Correlation Heatmap', fontsize=14)
    st.pyplot()

    st.sidebar.subheader("Bivariate relations")
    # Set up 2 columns to display in the body of the ap
    colbis1, colbis2 = st.sidebar.columns(2)
    V_1 = colbis1.selectbox('Target variable:',df.columns, index=0)
    V_2 = colbis2.selectbox('Key varaible:',df.columns, index=5)
    st.subheader("Visualising statistical relationships between two varaibles:")
    sns.set()
    sns.relplot(data=df, x=V_2, y=V_1, kind='line', height=5, aspect=2, color='red');
    st.pyplot()
    
    st.sidebar.subheader("Multivariate relations")
    # Set up 3 columns to display in the body of the ap
    colbis1, colbis2,colbis3 = st.sidebar.columns(3)
    V_3 = colbis1.selectbox('Target variable:',df.columns, index=2, key="Target variable_1")
    V_4 = colbis2.selectbox('Key varaible:',df.columns, index=3, key="Key varaible_2")
    V_5 = colbis3.selectbox('interact varaible:',df.columns, index=4)
    st.subheader("Visualising relationships between muti-varaibles:")
    sns.catplot(x=V_4, y=V_3, hue=V_5, kind="point", data=df)
    st.pyplot()
    


###############Creating Training and Test splits for Classification model###############
@st.cache(persist=True)
def split(df):
    y = df['level']
    x = df[['ordered', 'age', 'speed', 'single', 'dist','income']]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(df)

###############Creating Training and Test splits for Gregression model###############
@st.cache(persist=True)
def split(df):
    Y = df['sat']
    X = df[['ordered', 'age', 'speed', 'single', 'dist','income']]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
    return X_train, X_test, Y_train, Y_test
X_train, X_test, Y_train, Y_test = split(df)

###############Evaluation Metrics###############
def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels= class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
class_names = ["Satisfied", "Unsatisfied"]

st.set_option('deprecation.showPyplotGlobalUse', False)

###############Training an SVM classifier###############
st.sidebar.subheader("Choose Model")
classifier = st.sidebar.selectbox("Model", ("Random Forest Regression", "Random Forest classification"))

###############Visualising SHAP Explantions###############
@st.cache(persist=True,suppress_st_warning=True)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

###############Training Random Forest classifier Hyperparameters###############
if classifier == "Random Forest classification":
    st.sidebar.subheader("Hyperparameters")
    n_estimators= st.sidebar.number_input("The number of trees in the forest (100~5000)", 100, 5000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("The maximum depth of tree (1~20)", 1, 20, step =1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    st.sidebar.header("SHAP Explanations")
    st.sidebar.subheader("SHAP single prediction Explanations")
    individual = st.sidebar.number_input("Select the desired record from the testing set for detailed explanation (0~864)",
                                         min_value=min(range(len(x_test))),
                                         max_value=max(range(len(x_test))))
    
    st.sidebar.subheader("SHAP single feature Explanations")
    feature_selector_4 = st.sidebar.selectbox('Main feature :',X_test.columns, index=0, key="feature_selector_4")
# Set up 2 columns to display in the body of the ap
    st.sidebar.subheader('SHAP interactive feature Explanations')
    colbis1, colbis2 = st.sidebar.columns(2)

    # Selectors for dependence plot
    feature_selector = colbis1.selectbox('Main feature :',x_test.columns, index=0)
    interaction_selector = colbis2.selectbox('Interaction feature :',x_test.columns, index=5)

###############Training Random Forest classifier model###############
    if st.sidebar.button("Run", key="classify_2"):
        st.subheader("Random Forest Classification Model Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap= bootstrap, n_jobs=-1 )
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

        st.subheader('Feature Importance - Classification model')
        with st.expander("See Note"):
            st.write("Feature Importance: Observe the contribution of features to the model prediction process by looking at their importance.")
        importances_1 = model.feature_importances_
        indices_1 = np.argsort(importances_1)
        features_1 = x_train.columns
        pl.title('Feature Importances')
        pl.barh(range(len(indices_1)), importances_1[indices_1], color='skyblue', align='center')
        pl.yticks(range(len(indices_1)), [features_1[i] for i in indices_1])
        pl.xlabel('Relative Importance')
        st.pyplot(bbox_inches='tight')
        pl.clf()
        
 #############explain model predictions by SHAP###############         
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
    #############local interpretiability###############
        st.header("Local Explainability")
        st.write('Individual prediction explanation.')
        expectation = explainer.expected_value
        shap_values_1 = explainer.shap_values(x_test.iloc[individual,:])
        st.write('The RF classification model predicted that the **satisfaction level** of this customer with the Fast Delivery company is (**1: Satisfied; 0: Unsatisfied**): '+str(y_pred[individual]))
        real_value_1 = y_test.iloc[individual]
        st.write('The **real satisfaction level** of this individual customer is: '+str(real_value_1))
        st.write('Detailed information about this single prediction:',x_test.iloc[individual,:])
        
        st.subheader("The Force Plot - Individual prediction")
        with st.expander("Additional notes"):
            st.write('Features in **red increased** the prediction, in **blue decreased** them.')
            st.write('The **base value** in the force plot shows the **average predicted customer satisfied probability** of this classification model.')
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values_1[1], x_test.iloc[individual,:]))
        

        st.subheader("Interactive Force Plot")
        st.write('Visualise the all test set predictions')
        with st.expander("Additional notes"):
            st.write('An interactive force plot could be produced by taking many force plot explanations together, rotating them 90 degrees and stacking them horizontally.'
                     ' This interactive force plot can explain the predictions of multiple instances in one plot.'
                     ' The Y-axis is the X-axis of the individual force plot. There are 865 data points in the X_test, so the X-axis has 865 observations.')
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], x_test), 400)


      #############Global interpretiability###############
        st.header("Global Explainability")

        st.subheader("Global Explanations - SHAP Feature Importance plot")
        with st.expander("Notes"):
            st.write("""
         SHAP feature importance is measured as the mean absolute Shapley values.
         For both satisfied and unsatisfied classes, the number of times customers used (returned to) the food delivery platform was the most important feature.
         The customer’s age was the second most important feature.""")
        pl.title('SHAP Feature Importance')
        shap.summary_plot(shap_values,x_test,plot_type="bar",show=False)
        st.pyplot(bbox_inches='tight')
        pl.clf()

        st.subheader("Global Explanations - SHAP Summary Plot")
        with st.expander("Additional notes"):
            st.write('The features are ordered according to their importance; and the horizontal location in this plot shows whether the effect of that value is associated with a higher or lower prediction.' 
                  ' The biggest difference of this summary plot with the regular feature importance plot is that it shows the positive and negative relationships of the predictors with the target variable.'
                 ' A low number of times customers use the food delivery platform reduces the predicted customer satisfaction probability. A large number of used times increases the predicted probability.'
             ' A low number of customers age increases the predicted customer satisification probability, a large number of customers age decreases the predicted probability.')
        pl.title('SHAP Summary Plot')
        shap.summary_plot(shap_values[1],x_test,show=False)
        st.pyplot(bbox_inches='tight')
        pl.clf()
       

        
        st.subheader("Global Explanations - The SHAP Dependence Plot")
        with st.expander("Additional notes"):
            st.write('The SHAP dependence plot shows the effect of a single feature across the whole dataset, and tells whether the relationship between the target and the variable is linear, monotonic, or more complex.'
                     ' When selecting features at the sidebar, note that the alglorithm automatically plots the selected feature, with the feature that'
             ' it most likely interacts with.')
        dependence_plot_2= shap.dependence_plot(feature_selector_4,
                                              shap_values[1],
                                              X_test,
                                              show=True)
        pl.title('SHAP Dependence Plot', fontsize=10)
        st.pyplot(dependence_plot_2)
        
        st.subheader("Global Explanations - The Interactive Dependence Plot")
        with st.expander("Additional notes"):
            st.write('SHAP dependence plots are similar to partial dependence plots, but account for the interaction effects present in the features.'
                     ' The vertical dispersion of SHAP values at a single feature value is driven by interaction effects, and another feature is chosen for coloring to highlight possible interactions.')
        dependence_plot= shap.dependence_plot(feature_selector,
                                              shap_values[1],
                                              x_test,
                                              interaction_index=interaction_selector,
                                              show=True)
        pl.title('Feature Interactive -Customer satisifaction', fontsize=10)
        st.pyplot(dependence_plot)
       
###############Training Random Forest Regression - Hyperparameters###############
if classifier == "Random Forest Regression":
    st.sidebar.subheader("Hyperparameters")
    n_estimators_1= st.sidebar.number_input("The number of trees in the forest (100~5000)", 100, 5000, step=10, key="n_estimators_1")
    max_depth_1 = st.sidebar.number_input("The maximum depth of tree (1~20)", 1, 20, step =1, key="max_depth_1")

    st.sidebar.header("SHAP Explanations")
    st.sidebar.subheader("SHAP single predicition Explanations")
    individual_1 = st.sidebar.number_input("Select the desired record from the testing set for detailed explanation (0~864)",
                                         min_value=min(range(len(X_test))),
                                         max_value=max(range(len(X_test))), key="individual_1")
    
    st.sidebar.subheader("SHAP single feature Explanations")
    feature_selector_3 = st.sidebar.selectbox('Main feature :',X_test.columns, index=0, key="feature_selector_3")
    
    # Set up 2 columns to display in the body of the ap
    st.sidebar.subheader('SHAP interactive feature Explanations')
    colbis1, colbis2 = st.sidebar.columns(2)

    # Selectors for interactive dependence plot
    feature_selector_1 = colbis1.selectbox('Main feature :',X_test.columns, index=0, key="feature_selector_1")
    interaction_selector_1 = colbis2.selectbox('Interaction feature :',X_test.columns, index=5, key="interaction_selector_1")

###############Training Random Forest Regression - modelling###############
    if st.sidebar.button("Run", key="classify_3"):
        st.subheader("Random Forest Regression Model Results")
        model_1 = RandomForestRegressor(n_estimators=n_estimators_1, max_depth=max_depth_1, n_jobs=-1 )
        model_1.fit(X_train, Y_train)
        accuracy = 100 - np.mean(100 * (abs(model_1.predict(X_test) - Y_test) / Y_test))
        Y_pred = model_1.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        
        st.write('Using the standard Random Forest importance plot feature')
        with st.expander("Additional notes"):
            st.write("Feature Importance: Observe the contribution of features to the model prediction process by looking at their importance.")
        importances = model_1.feature_importances_
        indices = np.argsort(importances)
        features = X_train.columns
        pl.title('Feature Importance - Regression model')
        pl.barh(range(len(indices)), importances[indices], color='steelblue', align='center')
        pl.yticks(range(len(indices)), [features[i] for i in indices])
        pl.xlabel('Relative Importance')
        st.pyplot(bbox_inches='tight')
        pl.clf()

            
        #############local interpretiability###############
        st.header("Local Explainability")
        explainer_1 = shap.TreeExplainer(model_1)
        shap_values_2 = explainer_1.shap_values(X_test)
        shap_values_3 = explainer_1.shap_values(X_test.iloc[individual_1,:])
        
        st.subheader('Individual prediction explanation.')
        expectation = explainer_1.expected_value
        st.write('single prediction - **predicted satisfication score**: '+str(Y_pred[individual_1]))
        real_value = Y_test.iloc[individual_1]
        st.write('The **real satisfaction score** given by this individual customer is: '+str(real_value))
        st.write('Detailed information about this single prediction:',X_test.iloc[individual_1,:])
        
        st.subheader("The Force Plot - single prediction")
        st.write('Features in red increased the prediction, in blue decreased them')
        st_shap(shap.force_plot(explainer_1.expected_value, shap_values_2[individual_1,:], X_test.iloc[individual_1,:]))

        st.subheader('The Interactive Force Plot')
        st.write('Visualise the all test set predictions')
        with st.expander("See Notes"):
             st.write('An interactive force plot could be produced by taking many force plot explanations together, rotating them 90 degrees and stacking them horizontally.'
                     ' This interactive force plot can explain the predictions of multiple instances in one plot.'
                      ' The Y-axis is the X-axis of the individual force plot. There are 865 data points in the X_test, so the X-axis has 865 observations.')
        st_shap(shap.force_plot(explainer_1.expected_value, shap_values_2, X_test), 400)

        #############explain model predictions by SHAP###############
        st.header("Global Explainability")

        #############Global interpretiability###############
        st.subheader("Global explanations - SHAP Feature Importance")
        with st.expander("See Notes"):
            st.write("""
         SHAP feature importance is measured as the mean absolute Shapley values.
         The number of times customers used (returned to) the food delivery platform was the most important feature.
         The customer’s age was the second most important feature.""")
        pl.title('SHAP Feature Importance')
        shap.summary_plot(shap_values_2,X_test,plot_type="bar",show=False)
        st.pyplot(bbox_inches='tight')
        pl.clf()
       
        st.subheader("Global Explanations - SHAP Summary Plot")
        with st.expander("Additional notes"):
            st.write('The SHAP Summary Plot  uses a density scatter plot of the SHAP values for each feature to determine how much each feature affects the model output for individuals in the test dataset.'
                     ' Features are sorted by the sum of the SHAP value magnitudes across all samples.'
                ' And the biggest difference of this summary plot with the regular feature importance plot is that it shows the positive and negative relationships of the predictors with the target variable.')
        shap.summary_plot(shap_values_2,X_test,show=False)
        st.pyplot(bbox_inches='tight')
        pl.clf()
   
        #############feature (interactive) dependence plot###############
        # SHAP dependence plot
        st.subheader("Global Explanations - The SHAP dependence Plot")
        with st.expander("Additional notes"):
            st.write('The SHAP dependence plot shows the effect of a single feature across the whole dataset, and tells whether the relationship between the target and the variable is linear, monotonic, or more complex.'
                     ' When selecting features at the sidebar, note that the alglorithm automatically plots the selected feature, with the feature that'
             ' it most likely interacts with.')
        dependence_plot_3= shap.dependence_plot(feature_selector_3,
                                              shap_values_2,
                                              X_test,
                                              show=True)
        pl.title('SHAP Dependence Plot', fontsize=10)
        st.pyplot(dependence_plot_3)
        
        st.subheader("Global Explanations - The interactive dependence Plot")
        with st.expander("Additional notes"):
            st.write('SHAP dependence plots are similar to partial dependence plots, but account for the interaction effects present in the features.'
                     ' The vertical dispersion of SHAP values at a single feature value is driven by interaction effects, and another feature is chosen for coloring to highlight possible interactions.')
        dependence_plot= shap.dependence_plot(feature_selector_1,
                                              shap_values_2,
                                              X_test,
                                              interaction_index=interaction_selector_1,
                                              show=True)
        pl.title('Feature Interactive -Customer satisifaction Score', fontsize=10)
        st.pyplot(dependence_plot)
    
        
        
