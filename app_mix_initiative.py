import streamlit as st
import shap
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split

############### Visualising SHAP Explanations ###############
@st.cache_data(persist="disk")
def st_shap(_plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{_plot.html()}</body>"
    components.html(shap_html, height=height)

############### Training Random Forest classifier Hyperparameters ###############
if classifier == "Random Forest classification":
    st.sidebar.subheader("3.1 Hyperparameters")
    n_estimators = st.sidebar.number_input("The number of trees in the forest (100~5000)", 100, 5000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("The maximum depth of tree (1~20)", 1, 20, step=1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    st.sidebar.header("4. SHAP Explanations")
    st.sidebar.subheader("4.1 SHAP individual prediction explanations")
    individual = st.sidebar.number_input("Select the desired record from the testing set for detailed explanation (0~864)",
                                         min_value=0,
                                         max_value=len(x_test) - 1)

    st.sidebar.subheader("4.2 SHAP feature global explanations")
    feature_selector_4 = st.sidebar.selectbox('Main feature :', x_test.columns, index=0, key="feature_selector_4")

    st.sidebar.subheader('SHAP interactive feature explanations')
    colbis1, colbis2 = st.sidebar.columns(2)

    # Selectors for dependence plot
    feature_selector = colbis1.selectbox('Main feature :', x_test.columns, index=0)
    interaction_selector = colbis2.selectbox('Interaction feature :', x_test.columns, index=5)

############### Training Random Forest classifier model ###############
    if st.sidebar.button("Run", key="classify_2"):
        st.header("3.2 Random Forest Classification Model Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, average='weighted').round(2))
        plot_metrics(metrics)

        st.subheader('Feature Importance - Classification model')
        st.write("Feature Importance: Observe the contribution of features to the model prediction process by looking at their importance.")
        importances_1 = model.feature_importances_
        indices_1 = np.argsort(importances_1)
        features_1 = x_train.columns
        plt.title('Feature Importances')
        plt.barh(range(len(indices_1)), importances_1[indices_1], color='skyblue', align='center')
        plt.yticks(range(len(indices_1)), [features_1[i] for i in indices_1])
        plt.xlabel('Relative Importance')
        fig = plt.gcf()
        st.pyplot(fig, bbox_inches='tight')
        plt.clf()

############# explain model predictions by SHAP ###############
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        
        ############# local interpretability ###############
        st.header("4.1 Local Explainability")
        st.subheader('Local Explainability - Individual prediction explanation.')
        expectation = explainer.expected_value
        shap_values_1 = explainer.shap_values(x_test.iloc[individual,:])
        st.write('The RF classification model predicted that the **satisfaction level** of this customer with the Fast Delivery company is (**1: Satisfied; 0: Unsatisfied**): ,' + str(y_pred[individual]))
        real_value_1 = y_test.iloc[individual]
        st.write('The **real satisfaction level** of this individual customer is: ' + str(real_value_1))
        st.write('Detailed information about this single prediction:', x_test.iloc[individual,:])
        
        st.subheader("1) The Force Plot - Individual prediction")
        st.write('Which features caused this specific prediction? features in **red increased** the prediction, in **blue decreased** them.')
        st.write('The **base value** in the force plot shows the **average predicted customer satisfied probability** of this classification model.')
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values_1[1], x_test.iloc[individual,:]))

        st.subheader("2) Interactive Force Plot")
        st.write('Visualise the **all test set predictions**')
        with st.expander("See Notes"):
            st.write('An interactive force plot could be produced by taking many individual force plot explanations together, rotating them 90 degrees and stacking them horizontally.'
                     ' This interactive force plot can explain the **predictions of multiple instances** in one plot.'
                     ' The Y-axis is the X-axis of the individual force plot. There are 865 data points in the X_test, so the X-axis has 865 observations.')
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], x_test), 400)

############# feature importance plot ###############
        st.subheader("2.2 Feature Importance")
        st.write('The feature importance plot shows the contribution of each feature to the model.')
        shap.summary_plot(shap_values, x_test, plot_type="bar")
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()

############# dependence plot ###############
        st.subheader("3) Global Explanations - The SHAP Dependence Plot")
        st.write("The SHAP dependence plot shows the effect of a single feature across the whole dataset, and tells whether the relationship between the target and the variable is linear, monotonic, or more complex.")
        dependence_plot_3 = shap.dependence_plot(feature_selector, shap_values, x_test, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()

        st.subheader("4) Global Explanations - The Interactive Dependence Plot")
        st.write("Developing a deeper understanding of the data using SHAP: Interaction effects")
        with st.expander("Additional information"):
            st.write('SHAP dependence plots are similar to partial dependence plots, but account for the interaction effects present in the features.'
                     ' The vertical dispersion of SHAP values at a single feature value is driven by interaction effects, and another feature is chosen for coloring to highlight possible interactions.')
        dependence_plot = shap.dependence_plot(feature_selector, shap_values, x_test, interaction_index=interaction_selector, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()
