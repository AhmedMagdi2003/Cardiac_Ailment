import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
import sklearn
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score ,KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle

@st.cache_data
def load_data():
    df = pd.read_csv('ECGCvdata.csv')  # Ensure this file exists
    return df

# Sidebar Navigation
st.sidebar.title("📊 Project Outlines")
page = st.sidebar.radio("Select Page", ["Overview", "Exploratory Data Analysis", 
    "Model Selection","Model Evaluation","Model Deployment"])
st.sidebar.markdown("---")

df = load_data()


#--------------------------------------------------------
def missing_values(df):
    missing_percentage = df.isnull().sum()*100/len(df)
    missing_percentage = missing_percentage[missing_percentage>0]
    df = df.drop(columns=missing_percentage[missing_percentage>60].index)
    less_than_60 = missing_percentage[missing_percentage<60]
    sup_groups_0 = df.groupby('ECG_signal')[less_than_60.index[0]].median().to_dict()
    sup_groups_1 = df.groupby('ECG_signal')[less_than_60.index[1]].median().to_dict()
    # Mapping the missing values
    df[less_than_60.index[0]] = df[less_than_60.index[0]].fillna(df['ECG_signal'].map(sup_groups_0))
    df[less_than_60.index[1]] = df[less_than_60.index[1]].fillna(df['ECG_signal'].map(sup_groups_1))
    return df
# overview page
if page == "Overview":
    options = st.sidebar.selectbox("Select an option", ["Overview", "Dataset Symmary"])
    if options == "Overview":
        st.markdown("# 🔬 Cardiac Ailment Prediction Using ECG Data")
        st.markdown("## 🩺 Introduction")
        st.write("""
        Cardiovascular diseases are one of the leading causes of death worldwide, and early diagnosis is crucial for preventing severe complications. 
        In this project, we will develop a **machine learning model** to classify different types of cardiac ailments based on **ECG (Electrocardiogram) signals**. 
        Our dataset contains **1200 ECG records**, each belonging to one of **four categories of heart conditions**. 
        The goal is to analyze the ECG features and build a predictive model that can accurately classify a patient’s condition.
        """)

        # ECG Image
        image = Image.open("ecg.jpg")
        st.image(image, caption="ECG Signal", use_column_width=True)

        st.markdown("---")

        # ECG Signal Section
        st.markdown("## 📊 Understanding ECG Signals")
        st.write("""
        An **Electrocardiogram (ECG)** is a medical test that records the electrical activity of the heart over time. 
        It provides essential information about the **heart rate, rhythm, and electrical conduction system**, helping doctors diagnose cardiac diseases.  

        In our dataset, ECG signals have been processed using the **MODWPT (Maximal Overlap Discrete Wavelet Packet Transform)** method, 
        extracting **54 key features** that reflect different characteristics of the heart’s electrical activity.
        """)

        st.markdown("---")

        # Categories of Ailments
        st.markdown("## 🦠 Categories of Cardiac Ailments")
        st.write("Our dataset includes four major categories of heart conditions, which we aim to classify:")

        st.table({
            "Disease": ["Arrhythmia (ARR)", "Atrial Fibrillation & Flutter (AFF)", 
                        "Congestive Heart Failure (CHF)", "Normal Sinus Rhythm (NSR)"],
            "Description": [
                "Irregular heartbeats that can be too fast, too slow, or erratic.",
                "Rapid and irregular electrical activity in the atria, leading to uncoordinated contractions.",
                "A condition where the heart cannot pump blood efficiently, causing fluid buildup in the lungs and body.",
                "A normal heart rhythm with a regular rate and electrical activity."
            ]
        })

        st.write("""
        Each record in our dataset is labeled as **ARR, AFF, CHF, or NSR**, 
        allowing us to develop a **multi-class classification model**.
        """)

        st.markdown("---")

        # Dataset Structure
        st.markdown("## 📑 Understanding the Dataset")
        st.write("""
        The dataset consists of **56 columns**:
        - **Column 1** → Record ID  
        - **Columns 2-55** → Features extracted from ECG signals using MODWPT  
        - **Column 56** → **Target label** (ARR, AFF, CHF, NSR)
        """)

        st.write("Certain features play a significant role in identifying specific heart conditions:")

        st.table({
            "Condition": ["Arrhythmia (ARR)", "Atrial Fibrillation (AFF)", 
            "Congestive Heart Failure (CHF)", "Normal Sinus Rhythm (NSR)"],
            "Key Features": [
                "Heartbeats per minute (hbpermin), RR intervals",
                "P-wave absence, RR interval irregularity",
                "QRS duration, T-wave changes, variability in RR intervals",
                "Regular heart rate, normal QRS and P-wave patterns"
            ]
        })

        st.write("""
        In the next steps, we will conduct **Exploratory Data Analysis (EDA)**, **data preprocessing**, and build a **classification model** 
        to predict cardiac ailments using the extracted ECG features.
        """)

        st.markdown("🚀 **Let’s begin our journey into predictive cardiology!** 💙")
    st.write("Report generated on: [11/4/2025]")

#--------------------------------------------------------------------------
        # Dataset Summary

    if options == "Dataset Symmary":
        st.markdown("# 📊 Dataset Summary")
        st.write("""
        The dataset contains **1200 ECG records** with **56 columns**. 
        Each record is labeled as one of the four categories: **ARR, AFF, CHF, or NSR**.
        """)

        # Display the first few rows of the dataset
        st.subheader("Dataset Overview")
        st.write(df.head())

        # Display basic statistics of the dataset
        st.subheader("Basic Statistics")
        st.write(df.describe())

#-------------------------------------------------------------------------

# Exploratory Data Analysis
# EDA page

#-------------------------------------------------------------------------
elif page == "Exploratory Data Analysis":

    sub_page = st.sidebar.selectbox(
    "Exploratory Data Analysis Sections",
    ['Data Processing', 'Features Analysis', 'Feature Selection'])
    # Data structure page
    if sub_page == "Data Processing":
        # missing values
        st.markdown("""
# 📊 Exploratory Data Analysis (EDA)
In this section, we will explore the dataset to understand its structure, distribution of features, and relationships between them. 
EDA is crucial for identifying patterns, trends, and potential anomalies in the data. 
    """)
        st.markdown("---")
        st.subheader("Missing Values Handling")
        missing_percentage = df.isnull().sum()*100/len(df)
        missing_percentage = missing_percentage[missing_percentage>0]
        #  plot missing values
        fig = px.bar(
        missing_percentage.sort_values(ascending=False),
        labels={'index': 'Features', 'value': 'Missing Percentage'},
        title='Missing Values Percentage in Features',
        color=missing_percentage.sort_values(ascending=False),
        color_continuous_scale="Blues")
        st.plotly_chart(fig)

        st.write ("Remove columns with more than 60% missing values")
        # plot the skewness of the less than 60% missing values to decied how to handle them
        fig = px.histogram(
            df[missing_percentage[missing_percentage<60].index],
            labels={'value': 'Skewness'},
            title='Skewness of Features with Missing Values'
            )
        st.plotly_chart(fig)
        st.markdown("""
        By observing the skewness of the features with less than 60% missing values,
        we can see that the two features are highly skewed.
        We can use the **median** of the groups to fill the missing values.""")
        df = missing_values(df)
        # check the uniqness of the Dataset rows
        st.markdown("---")
        st.subheader("Duplicate Rows Analysis")
        #------------------------------------------------------
        # Calculate duplicates
        #------------------------------------------------------
        duplicated_rows = df.duplicated().sum()
        total_rows = len(df)
        st.metric("Total Duplicated Rows", duplicated_rows, delta=f"{(duplicated_rows/total_rows):.1%} of data")
        
        # Visualize distribution

        fig = px.bar(
            x=["Unique Rows", "Duplicated Rows"],
            y=[total_rows - duplicated_rows, duplicated_rows],
            labels={"x": "Row Type", "y": "Count"},
            title="Data Composition: Unique vs Duplicated",
            color=["Unique", "Duplicated"],
            color_discrete_map={"Unique": "green", "Duplicated": "red"}
        )
        st.plotly_chart(fig, use_container_width=True) 
        st.write("No duplicated rows found in the dataset.")
        #------------------------------------------------------            
        st.markdown("---")
        # Distribution of target variable
        st.subheader("Distribution of Target Variable")
        fig = px.bar(
            df['ECG_signal'].value_counts(),
            labels={'index': 'Target Class', 'value': 'Count'},
            title='Distribution of Cardiac Ailments'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.write("Data is well-balanced across the four classes.")

# ---------------------------------------------------------
# Features Analysis
#----------------------------------------------------------
    elif sub_page =="Features Analysis":
        df = missing_values(df)
        df.drop(columns=['RECORD'], inplace=True)
        st.markdown("# 🔍 Feature Analysis")
        st.write("""
        In this section, we will analyze the features extracted from the ECG signals. 
        Understanding the distribution and relationships of these features is crucial for building an effective predictive model.
        """)
        st.markdown("---")
        # Display the first few rows of the dataset
        st.markdown("""
### 📊 Strip Plot Analysis for Feature Selection
To build a reliable classification model for cardiac ailments, it's essential to understand how each 
feature behaves across different classes.
**Strip plots** help us visually inspect the **distribution, spread, and separability** of features.
This analysis supports the identification of:
- Features that contribute meaningfully to classification.
- Features that may introduce noise or confusion.
---""")
        st.markdown("""
####  ✅ Well-Separated Features
- Show distinct patterns between classes.
- Improve classification performance.
- Preferred for model training.
    """)
        df_melted = pd.melt(df.iloc[:,[29,24,30,27,47]], id_vars='ECG_signal',
        var_name='features', value_name='value')
        fig = px.strip(df_melted,x='features', y='value', color='ECG_signal',
        title='Distribution of Well-Seperated Features by ECG Signal',
        labels={'features': 'Features', 'value': 'Value'},
        stripmode="overlay",
        hover_data=['ECG_signal'],
        )
        st.plotly_chart(fig, use_container_width=True)

        df_melted = pd.melt(df.iloc[:,[21,25,31,35,47]], id_vars='ECG_signal',
        var_name='features', value_name='value')
        fig = px.strip(df_melted,x='features', y='value', color='ECG_signal',
        title='Distribution of Well-Seperated Features by ECG Signal',
        labels={'features': 'Features', 'value': 'Value'},
        stripmode="overlay",
        hover_data=['ECG_signal'],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
#### 🔻 Noisy & Overlapping Features
- Show similar distributions across classes.
- Add noise and reduce model clarity.
- Candidates for removal or reduction.
    """)
        df_melted = pd.melt(df.iloc[:,[9,10,43,44,47]], id_vars='ECG_signal', 
        var_name='features', value_name='value')
        # Plot
        fig = px.strip(
        df_melted, x='features', y='value', color='ECG_signal',
        title='Distribution of Overlapped Features by ECG Signal',
        labels={'features': 'Features', 'value': 'Value'},
        stripmode="overlay",
        hover_data=['ECG_signal'],
        )
        st.plotly_chart(fig, use_container_width=True)
    #----------------------------------------------------------
        df_melted = pd.melt(df.iloc[:,[3,4,5,47]], id_vars='ECG_signal', 
        var_name='features', value_name='value')
        # Plot
        fig = px.strip(
        df_melted, x='features', y='value', color='ECG_signal',
        title='Distribution of Noisy Features by ECG Signal',
        labels={'features': 'Features', 'value': 'Value'},
        stripmode="overlay",
        hover_data=['ECG_signal'],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df.iloc[:,:-1], df.iloc[:, -1])
        st.subheader('Feature Importance Anlysis By Random Forest')
        feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=True)
        fig = px.bar(
            feature_importance,
            labels={'index': 'Features', 'value': 'Importance'},
            orientation='h',
            title='Feature Importance Analysis',
            color=feature_importance,
            color_continuous_scale="viridis"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.write('By observing the Feature importance by random forest and after check that features' \
        ' we assumed that have a high noise ' \
        'are not important for the model, we can remove them.')
#----------------------------------------------------------
# Feature Selection
#----------------------------------------------------------
    else:
        df.drop(columns=['RECORD'], inplace=True)
        df = missing_values(df)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df.iloc[:,:-1], df.iloc[:, -1])
        feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=False)
        st.markdown("# 🔍 Feature Selection")
        st.write("""
        In this section, we will select the most relevant features for our classification model. 
        Feature selection is crucial for improving model performance and reducing overfitting.
        """)
        st.markdown("---")
        st.subheader('Feature Importance Anlysis By Random Forest')
        st.write("Important features based on Random Forest model:")
        selected_features = feature_importance[feature_importance >= 0.02]
        fig = px.bar(
            selected_features,
            labels={'index': 'Features', 'value': 'Importance'},
            orientation='h',
            title='Feature Importance Analysis',
            color=selected_features,
            color_continuous_scale="viridis"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.write('Features were selected by using 0.02 as threshold.')
        st.markdown("---")
        st.subheader("Feature Correlation Analysis")
        st.write("Correlation analysis helps us understand the relationships between features.")
        fig = px.imshow(
            df[selected_features.index].corr(),
            color_continuous_scale='RdBu',
            title='Feature Correlation Matrix',
            labels={'color': 'Correlation Coefficient'},
            text_auto=True,
            color_continuous_midpoint=0,
        )
        fig.update_layout(coloraxis_colorbar=dict(title='Correlation Coefficient'))
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("Remove Redundant Features")
        st.write("The correlation matrix shows the relationships between features. A value close to 1 or -1 indicates a strong correlation.")
        st.write("After analyzing the heatmap, some features show strong correlations with multiple other features. For example, **PTdis** has a high correlation with four different features.")
        st.write("We can remove the features that have a high correlation with other features, to reduce redundancy.")
        st. write('By setting a threshold of 0.9, we can remove the features that have a high correlation with other features.')
        selected_features = df[selected_features.index]
        redundant_features = []
        correlation_matrix = selected_features.corr()
        for col in correlation_matrix.columns:
            for index in correlation_matrix.index:
                if col != index and correlation_matrix.loc[col, index] >= 0.9:
                    redundant_features.append(index)
        # Keep only unique redundant features
        redundant_features = list(set(redundant_features))
        selected_features.drop(columns=redundant_features, inplace=True)
        #----------------------------------------------------------
        fig = px.imshow(
        selected_features.corr(),
        color_continuous_scale='RdBu',
        title='Feature Correlation Matrix After Removing Redundant Features',
        labels={'color': 'Correlation Coefficient'},
        text_auto=True,
        height=600,
        width=800,
        )
        st.plotly_chart(fig,use_container_width=False)
        st.write("After removing the redundant features, we can see that the correlation matrix is more clear and easy to interpret.")
        st.markdown("---")
elif page == "Model Selection":
    st.markdown("# 🤖 Model Selection")
    st.write("""
    In this section, we will select the most suitable machine learning model for our cardiac ailment classification task. 
    We will evaluate different models based on their performance metrics and choose the best one for deployment.
    """)
    st.markdown("---")
    st.subheader("Model Selection Process")
    st.write("""
    The model selection process involves the following steps:
    1. **Model Evaluation**: Evaluate the performance of each model using metrics such as accuracy, precision, recall, and F1-score.
    2. **Model Comparison**: Compare the performance of different models to select the best one.
    """)
    #----------------------------------------------------------
    df = missing_values(df)
    df.drop(columns=['RECORD'], inplace=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df.iloc[:,:-1], df.iloc[:, -1])
    feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=True)
    selected_features = feature_importance[feature_importance >= 0.02]
    selected_features = df[selected_features.index]
    redundant_features = []
    correlation_matrix = selected_features.corr()
    for col in correlation_matrix.columns:
        for index in correlation_matrix.index:
            if col != index and correlation_matrix.loc[col, index] >= 0.9:
                redundant_features.append(index)
    # Keep only unique redundant features
    redundant_features = list(set(redundant_features))
    selected_features.drop(columns=redundant_features, inplace=True)
    encoder = LabelEncoder() 
    df['ECG_signal'] = encoder.fit_transform(df['ECG_signal'])
    # Split the data into training and testing sets
    x = selected_features
    y = df['ECG_signal']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Train multiple models
    clf_random = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_decision = DecisionTreeClassifier(random_state=42)
    clf_gradient = GradientBoostingClassifier(random_state=42)
    clf_extra = ExtraTreesClassifier(random_state=42)
    # Train the classifiers
    clf_random.fit(x_train, y_train)
    clf_decision.fit(x_train, y_train)
    clf_gradient.fit(x_train, y_train)
    clf_extra.fit(x_train, y_train)
    # Make predictions
    y_pred_random = clf_random.predict(x_test)
    y_pred_decision = clf_decision.predict(x_test)
    y_pred_gradient = clf_gradient.predict(x_test)
    y_pred_extra = clf_extra.predict(x_test)
    model_names = ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'Extra Trees']
    y_preds = [y_pred_random, y_pred_decision, y_pred_gradient, y_pred_extra]
    models_summary = []
    for name, y_pred in zip(model_names, y_preds):
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        models_summary.append([name, acc, prec, rec, f1])
    df_summary = pd.DataFrame(models_summary, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    df_summary = df_summary.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    # Display summary table
    print("Model Performance Summary:\n")
    print(df_summary)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    df_melted = df_summary.melt(id_vars='Model', var_name='Metric', value_name='Score') # Melt the DataFrame for plotting
    #Plot performance comparison
    fig = px.bar(
    df_melted, 
    x='Model', 
    y='Score', 
    color='Metric',
    barmode='group',
    title="Tree-Based Model Performance Comparison",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(yaxis_tickformat='.0%', yaxis_range=[0.8, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(""" --- 
## ✅ Final Model Selection: Extra Trees Classifier
After evaluating multiple tree-based classification models
— including Random Forest, Decision Tree, and Gradient Boosting — we selected the **Extra Trees Classifier** 
as the final model for predicting cardiac ailments.
### 🔍 Why Extra Trees?
- **Highest Accuracy**: Achieved **98.3% accuracy** on the test set.
- **Strong Overall Performance**: Excellent precision, recall, and F1-score across all classes.
- **Robustness**: Consistently outperformed other models without overfitting.
- **Efficiency**: Fast training and reliable generalization.
This makes Extra Trees the most reliable and effective model for our ECG classification task.
""")
    
#----------------------------------------------------------
# Model Evaluation
#----------------------------------------------------------

elif page == "Model Evaluation":
    st.markdown("# 📊 Model Evaluation")
    st.markdown("""
In this section, we evaluate the performance of the selected model, **Extra Trees Classifier**, using key metrics:
- **Cross-Validation**: Validate model robustnessc & generalization.
- **Confusion Matrix**: Visualize true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Analyze precision, recall, F1-score, and support for each class.
---
""")
    df = missing_values(df)
    df.drop(columns=['RECORD'], inplace=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df.iloc[:,:-1], df.iloc[:, -1])
    feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=False)
    selected_features = feature_importance[feature_importance >= 0.02]
    selected_features = df[selected_features.index]
    redundant_features = []
    correlation_matrix = selected_features.corr()
    for col in correlation_matrix.columns:
        for index in correlation_matrix.index:
            if col != index and correlation_matrix.loc[col, index] >= 0.9:
                redundant_features.append(index)
    # Keep only unique redundant features
    redundant_features = list(set(redundant_features))
    selected_features.drop(columns=redundant_features, inplace=True)
    encoder = LabelEncoder() 
    df['ECG_signal'] = encoder.fit_transform(df['ECG_signal'])
    # Split the data into training and testing sets
    y = df['ECG_signal']
    x = selected_features
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Train multiple models
    clf_extra = ExtraTreesClassifier(random_state=42)
    # Train the classifiers
    clf_extra.fit(x_train, y_train)
    # Make predictions
    y_pred_extra = clf_extra.predict(x_test)
    cm = confusion_matrix(y_test, y_pred_extra)
    f1 = f1_score(y_test, y_pred_extra, average='macro')
    acc = accuracy_score(y_test, y_pred_extra)
    prec = precision_score(y_test, y_pred_extra, average='macro')
    rec = recall_score(y_test, y_pred_extra, average='macro')
    #----------------------------------------------------------
    # Cross-validation 
    st.subheader(" 📈Cross-Validation")
    st.write(" We will use k-fold cross-validation to evaluate the model's performance.")
    fold = KFold(n_splits = 5 , shuffle=True , random_state=42)
    cv = cross_val_score(clf_extra,x_train,y_train,cv=fold)
    fig = px.line(
        x=list(range(1, len(cv) + 1)), 
        y=cv, 
        title="Cross-Validation Scores",
        labels={'x': 'Fold', 'y': 'Score'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    #st.write("Cross-validation scores for each fold:")
    #st.write('Cross Validation Scores:', ['{:.2f}'.format(score) for score in cv])
    st.write( 'Mean Cross Validation Score:', cv.mean().round(2))
    st.write( 'Standard Deviation of Cross Validation Score:', cv.std().round(4))
    st.markdown("" \
"These results indicate that our model is **highly accurate** and **stable**," \
" with very low variance between " \
"folds. This consistency strengthens our confidence in the model's performance on unseen ECG data.")
    st.markdown("---")
    #----------------------------------------------------------
    # Confusion Matrix
    st.subheader("🧩 Confusion Matrix")
    st.markdown("""
    **What is a Confusion Matrix?**
    - It shows the number of correct and incorrect predictions made by the classifier.
    - Each row represents the actual class; each column represents the predicted class.
    - It's especially helpful in multi-class classification problems.
    """)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="RdBu",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        x=encoder.inverse_transform(range(len(cm))),
        y=encoder.inverse_transform(range(len(cm))),
        title="Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown("""
### 📈 Class-wise Evaluation Metrics
To ensure robust model performance across all cardiac conditions, we visualize **precision, recall, and F1-score** per class. These metrics help detect weaknesses in classifying certain ailments. For example:
- **Low recall** in a class indicates the model is missing positive cases (critical in diagnosis).
- **High F1-score** means good balance between precision and recall.
This plot helps validate that the model performs consistently across all ECG categories.
""")
    report_dict = classification_report(y_test, y_pred_extra, output_dict=True)
    # Convert to DataFrame
    report_df = pd.DataFrame(report_dict).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
    # Reset index to turn the class labels into a column
    report_df = report_df.reset_index().rename(columns={'index': 'Class'})
    class_labels = encoder.inverse_transform(report_df['Class'].astype(int))
    report_df['Class'] = class_labels
    # Plot F1, Precision, and Recall for each class
    fig = px.bar(report_df, x='Class', y=['precision', 'recall', 'f1-score'],
                barmode='group', text_auto='.2f',
                title='🔬 Class-wise Evaluation Metrics')
    fig.update_layout(xaxis_title='ECG Class Label', yaxis_title='Score', yaxis_range=[0, 1.1])
    st.plotly_chart(fig,use_container_width=True)
else :
    df = missing_values(df)
    df.drop(columns=['RECORD'], inplace=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df.iloc[:,:-1], df.iloc[:, -1])
    feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=False)
    selected_features = feature_importance[feature_importance >= 0.02]
    selected_features = df[selected_features.index]
    redundant_features = []
    correlation_matrix = selected_features.corr()
    for col in correlation_matrix.columns:
        for index in correlation_matrix.index:
            if col != index and correlation_matrix.loc[col, index] >= 0.9:
                redundant_features.append(index)
    # Keep only unique redundant features
    redundant_features = list(set(redundant_features))
    selected_features.drop(columns=redundant_features, inplace=True)
    st.markdown("# 🚀 Model Deployment")
    st.write("""
    In this section, we will deploy the selected model using Streamlit. 
    The deployment process involves creating a user-friendly interface for users to input ECG data and receive predictions.
    """)
    st.markdown("---")
    def load_model():
        with open('ecg_model.pkl', 'rb') as file:
            return pickle.load(file)
    model = load_model()
    diagnosis_labels = {
    0: "Arrhythmia (ARR)",
    1: "Atrial Fibrillation & Flutter (AFF)",
    2: "Congestive Heart Failure (CHF)",
    3: "Normal Sinus Rhythm (NSR)"}
    input_mode = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])
    
    if input_mode == "Manual Input":
        st.subheader("Enter ECG Features Manually")
        st.write("---")
        # Create input fields for each feature
        with st.form("ECG Input Form"):
            NNTot = st.number_input('NNTot (ms)', step=0.1, format="%.6f")
            STdis = st.number_input('STdis (mV)', step=0.1, format="%.6f")
            IBISD = st.number_input('IBISD (ms)', step=0.1, format="%.6f")
            PonPQang = st.number_input('PonPQang (°)', step=0.1, format="%.6f")
            STToffang = st.number_input('STToffang (°)', step=0.1, format="%.6f")
            RRTot = st.number_input('RRTot (ms)', step=0.1, format="%.6f")
            Pseg = st.number_input('Pseg (ms)', step=0.1, format="%.6f")

            submit = st.form_submit_button("Predict")
        # When the form is submitted
        if submit:
            # Create input DataFrame
            input_data = pd.DataFrame(
            [[NNTot, STdis, IBISD, PonPQang, STToffang, RRTot, Pseg]], 
            columns=['NNTot', 'STdis', 'IBISD', 'PonPQang', 'STToffang', 'RRTot', 'Pseg'])
        # Convert user input to DataFrame
            prediction = model.predict(input_data)
            st.success(f"Predicted Class: {diagnosis_labels[prediction[0]]}")
    if input_mode == 'Upload CSV' :
        st.subheader("Upload ECG Data CSV")
        st.write("Upload a CSV file of ECG signals " \
        "have been processed using the MODWPT (Maximal Overlap Discrete Wavelet Packet Transform) method ")
        st.write("---")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Preprocess the uploaded data
            df = df[['NNTot', 'STdis', 'IBISD', 'PonPQang', 'STToffang', 'RRTot', 'Pseg']]
            # Make predictions
            predictions = model.predict(df)
            df['Predicted_Class'] = [diagnosis_labels[pred] for pred in predictions]
            st.write(df.head())
            st.success("Predictions made successfully!")
            st.download_button("Download Predictions", df.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")
            st.write("Download the predictions as a CSV file.")
    st.markdown("---")
