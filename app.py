# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Page Title
st.title("Crimes on Women Dataset - Logistic Regression App")
st.write("🚀 This app trains a Logistic Regression model on the Crimes on Women dataset and provides visual insights.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CrimesOnWomenData.csv", type=['csv'])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Preprocessing
    df = df.drop(columns=['Unnamed: 0'])
    
    X = df.drop(['DV'], axis=1)
    y = (df['DV'] > 0).astype(int)
    
    # One-hot encode 'State'
    X = pd.get_dummies(X, columns=['State'], drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Show Model Accuracy
    st.subheader("Model Performance")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_mat = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['No DV', 'DV'], yticklabels=['No DV', 'DV'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig1)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(fpr, tpr, color='darkorange', lw=3, linestyle='-', label='ROC Curve (AUC = %0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    
    
    
    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    st.success("✅ Model training & visualization completed!")

else:
    st.warning("Please upload your CrimesOnWomenData.csv to continue.")
