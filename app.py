import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Crimes on Women Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .danger-card {
        background: linear-gradient(45deg, #ff416c 0%, #ff4b2b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# File paths for model persistence
MODEL_PATH = "crimes_model.pkl"
SCALER_PATH = "crimes_scaler.pkl"
FEATURES_PATH = "crimes_features.pkl"

def save_model_artifacts(model, scaler, feature_names):
    """Save trained model, scaler, and feature names"""
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(feature_names, f)
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def load_model_artifacts():
    """Load trained model, scaler, and feature names"""
    try:
        if all(os.path.exists(path) for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            with open(FEATURES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
            return model, scaler, feature_names
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def create_sample_data():
    """Create sample data for demonstration if no dataset is uploaded"""
    np.random.seed(42)
    
    states = ['Uttar Pradesh', 'Maharashtra', 'West Bengal', 'Rajasthan', 'Bihar', 
              'Madhya Pradesh', 'Tamil Nadu', 'Karnataka', 'Gujarat', 'Andhra Pradesh']
    
    data = []
    for i in range(1000):
        state = np.random.choice(states)
        
        # Generate correlated crime data
        base_crime_rate = np.random.uniform(0.1, 0.8)
        
        # Crime types with some correlation
        rape = max(0, int(np.random.poisson(base_crime_rate * 50)))
        kidnapping = max(0, int(np.random.poisson(base_crime_rate * 30)))
        dowry_deaths = max(0, int(np.random.poisson(base_crime_rate * 10)))
        assault = max(0, int(np.random.poisson(base_crime_rate * 80)))
        
        # Domestic Violence (target) - influenced by other crimes
        dv_prob = (rape * 0.3 + kidnapping * 0.2 + dowry_deaths * 0.4 + assault * 0.1) / 100
        dv_prob = min(0.9, max(0.1, dv_prob + np.random.normal(0, 0.2)))
        dv = int(np.random.binomial(1, dv_prob) * np.random.poisson(20))
        
        data.append({
            'State': state,
            'Rape': rape,
            'Kidnapping and Abduction': kidnapping,
            'Dowry Deaths': dowry_deaths,
            'Assault on women': assault,
            'DV': dv
        })
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the dataset"""
    # Remove unnamed columns if they exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Create binary target
    X = df.drop(['DV'], axis=1)
    y = (df['DV'] > 0).astype(int)
    
    # One-hot encode 'State' if it exists
    if 'State' in X.columns:
        X = pd.get_dummies(X, columns=['State'], drop_first=True)
    
    return X, y

def train_model(X, y):
    """Train the logistic regression model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'confusion_matrix': conf_mat,
        'classification_report': classification_report(y_test, y_pred),
        'roc_data': (fpr, tpr, roc_auc),
        'feature_names': X.columns.tolist()
    }

def display_model_performance(results):
    """Display model performance metrics"""
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h2>{results['accuracy']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>AUC Score</h3>
            <h2>{results['roc_data'][2]:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Features</h3>
            <h2>{len(results['feature_names'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Confusion Matrix")
        fig_conf = px.imshow(
            results['confusion_matrix'],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No DV', 'DV Reported'],
            y=['No DV', 'DV Reported'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_conf.update_layout(height=400)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        st.subheader("📈 ROC Curve")
        fpr, tpr, roc_auc = results['roc_data']
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            title='ROC Curve',
            height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Classification Report
    with st.expander("📋 Detailed Classification Report"):
        st.text(results['classification_report'])

def prediction_page():
    """Prediction page interface"""
    st.markdown('<h1 class="main-header">🔮 Make Predictions</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model_artifacts()
    
    if model is None:
        st.error("❌ No trained model found! Please train a model first in the 'Train Model' tab.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create input form
    st.subheader("📝 Enter Crime Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Core Crime Statistics:**")
        rape = st.number_input("Rape Cases", min_value=0, max_value=1000, value=10)
        kidnapping = st.number_input("Kidnapping & Abduction Cases", min_value=0, max_value=1000, value=15)
        dowry_deaths = st.number_input("Dowry Deaths", min_value=0, max_value=100, value=2)
        assault = st.number_input("Assault on Women Cases", min_value=0, max_value=1000, value=25)
    
    with col2:
        st.write("**Location Information:**")
        # Get state features from feature names
        state_features = [f for f in feature_names if f.startswith('State_')]
        
        if state_features:
            available_states = [f.replace('State_', '') for f in state_features]
            selected_state = st.selectbox("Select State", ['Other'] + available_states)
        else:
            selected_state = None
            st.info("No state information available in the model")
    
    # Make prediction
    if st.button("🎯 Predict Domestic Violence Risk", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'Rape': rape,
                'Kidnapping and Abduction': kidnapping,
                'Dowry Deaths': dowry_deaths,
                'Assault on women': assault
            }
            
            # Add state features
            for feature in state_features:
                state_name = feature.replace('State_', '')
                input_data[feature] = 1 if selected_state == state_name else 0
            
            # Create DataFrame with all features
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features are present
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Reorder columns to match training data
            input_df = input_df[feature_names]
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            risk_probability = prediction_proba[1] * 100
            
            if prediction == 1:
                st.markdown(f"""
                <div class="danger-card">
                    <h2>⚠️ HIGH RISK AREA</h2>
                    <h3>Domestic Violence likely to be reported</h3>
                    <h3>Risk Probability: {risk_probability:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>✅ LOW RISK AREA</h2>
                    <h3>Domestic Violence less likely to be reported</h3>
                    <h3>Risk Probability: {risk_probability:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk factors analysis
            st.subheader("📊 Risk Factors Analysis")
            
            # Create risk score breakdown
            risk_factors = {
                'Rape Cases': (rape / 50) * 100 if rape > 0 else 0,
                'Kidnapping Cases': (kidnapping / 30) * 100 if kidnapping > 0 else 0,
                'Dowry Deaths': (dowry_deaths / 5) * 100 if dowry_deaths > 0 else 0,
                'Assault Cases': (assault / 80) * 100 if assault > 0 else 0
            }
            
            # Normalize to 100
            total_risk = sum(risk_factors.values())
            if total_risk > 0:
                risk_factors = {k: min(100, (v/total_risk)*100*4) for k, v in risk_factors.items()}
            
            fig_risk = go.Figure(data=[
                go.Bar(
                    x=list(risk_factors.keys()),
                    y=list(risk_factors.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                    text=[f'{v:.1f}%' for v in risk_factors.values()],
                    textposition='auto'
                )
            ])
            
            fig_risk.update_layout(
                title="Individual Risk Factor Contribution",
                xaxis_title="Crime Types",
                yaxis_title="Risk Contribution (%)",
                height=400
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def main():
    # Sidebar navigation
    st.sidebar.title("⚖️ Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Train Model", "Make Predictions", "About"])
    
    if page == "Train Model":
        st.markdown('<h1 class="main-header">🏛️ Crimes on Women - ML Training</h1>', unsafe_allow_html=True)
        
        # Check if model already exists
        model, scaler, feature_names = load_model_artifacts()
        
        if model is not None:
            st.success("✅ Trained model found! You can directly go to 'Make Predictions' or retrain below.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔮 Go to Predictions", type="primary"):
                    st.experimental_rerun()
            with col2:
                if st.button("🔄 Retrain Model"):
                    # Delete existing model files
                    for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
                        if os.path.exists(path):
                            os.remove(path)
                    st.experimental_rerun()
        
        # File upload section
        st.subheader("📁 Dataset Upload")
        uploaded_file = st.file_uploader("Upload CrimesOnWomenData.csv (Optional - Sample data will be used if not provided)", type=['csv'])
        
        use_sample = st.checkbox("Use sample dataset for demonstration", value=True)
        
        if uploaded_file is not None or use_sample:
            # Load data
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.info("Using uploaded dataset")
            else:
                df = create_sample_data()
                st.info("Using sample dataset for demonstration")
            
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head())
            
            # Dataset statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns) - 1)
            with col3:
                dv_cases = (df['DV'] > 0).sum()
                st.metric("DV Cases", dv_cases)
            
            # Train model button
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                with st.spinner("Training model... Please wait"):
                    try:
                        X, y = preprocess_data(df)
                        results = train_model(X, y)
                        
                        # Save model
                        if save_model_artifacts(results['model'], results['scaler'], results['feature_names']):
                            st.success("✅ Model trained and saved successfully!")
                            display_model_performance(results)
                        else:
                            st.error("❌ Model training completed but failed to save")
                        
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
        
        else:
            st.warning("Please upload a dataset or check 'Use sample dataset' to continue.")
    
    elif page == "Make Predictions":
        prediction_page()
    
    else:  # About page
        st.markdown('<h1 class="main-header">ℹ️ About This App</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 Purpose
        This application predicts the likelihood of domestic violence cases based on other crime statistics in different regions.
        
        ## 🔧 Features
        - **Model Persistence**: Trained models are saved and can be reused
        - **Sample Data**: Demo functionality without requiring dataset upload
        - **Interactive Predictions**: Easy-to-use prediction interface
        - **Visual Analytics**: Comprehensive performance metrics and visualizations
        
        ## 📊 Model Details
        - **Algorithm**: Logistic Regression
        - **Features**: Rape cases, Kidnapping & Abduction, Dowry Deaths, Assault cases, State information
        - **Target**: Binary classification (DV cases reported vs not reported)
        
        ## 🚀 How to Use
        1. **Train Model**: Upload dataset or use sample data to train the model
        2. **Make Predictions**: Use the trained model to predict DV risk for new data
        3. **View Results**: Get probability scores and risk factor analysis
        
        ## 📈 Model Performance
        The model provides accuracy metrics, confusion matrix, ROC curve, and detailed classification reports.
        
        ## ⚠️ Important Note
        This is a demonstration tool. For real-world applications, ensure you have appropriate data permissions and consider ethical implications.
        """)

if __name__ == "__main__":
    main()
