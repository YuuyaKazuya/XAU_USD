import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained models
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page layout to wide
st.set_page_config(layout="wide")

# Title and Description with clickable link
st.title("📊 One-Day-Ahead XAU/USD Price Movement Prediction")
st.markdown("""
This dashboard forecasts the XAU/USD exchange rate trends using three different models:  
- **LightGBM**
- **Random Forest**
- **SVM**

The models predict future price movements based on historical data and technical indicators.

For historical forex data, visit [ForexSB](https://forexsb.com/historical-forex-data).
""")

# User Guide Section with collapsible expander
with st.expander("📖 User Guide"):
    st.markdown("""
    ### How to Use the Dashboard:

    1. **Upload Your Dataset**:
       - Click on the **"Upload CSV File"** button on the sidebar to upload your historical XAU/USD dataset.
       - The file should contain the following columns: Date, Open, High, Low, Close, Volume.

    2. **Generate Technical Indicators**:
       - Once the dataset is uploaded, click the **"Generate Technical Indicators"** button to calculate key technical indicators like SMA, WMA, RSI, MACD, and more. This step will add these indicators to your dataset.

    3. **View Cleaned Data**:
       - After generating the technical indicators, you will see the cleaned data along with the newly added columns of technical indicators.

    4. **Run Forecast**:
       - To predict the price movement, click the **"Run Forecast"** button on the sidebar.
       - The models will predict the trend of the XAU/USD price using the generated technical indicators.
       - You can view the prediction results for **LightGBM**, **Random Forest**, and **SVM** models, along with their accuracy.

    5. **Visualize Predictions**:
       - View the comparison between **actual** and **predicted** trends for each model in the interactive chart below.

    ### How to Interpret Results:
    - **Trend (Up/Down)**: 
       - If the **"Trend"** value is **1**, it means the price movement is predicted to go **up**.
       - If the **"Trend"** value is **-1**, it means the price movement is predicted to go **down**.
       
    - **Accuracy**:
       - The **"Correct"** column next to each model indicates whether the model correctly predicted the trend for that day: **✔️** for correct and **❌** for incorrect.

    For any issues, check the error messages shown above the chart or dataset.
    """)

# Load the pre-trained models
lgb_model_path = "best_lgb_discrete.pkl"
rf_model_path = "best_rf_discrete.pkl"
svm_model_path = "best_svm_discrete.pkl"

lgb_model = joblib.load(lgb_model_path)
rf_model = joblib.load(rf_model_path)
svm_model = joblib.load(svm_model_path)

st.success("✅ All Pre-trained Models Loaded")

# Sidebar
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv"])

# Display uploaded data
if uploaded_file:
    df1 = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.subheader("First Rows of the Dataset")
    st.dataframe(df1.head())
    st.subheader("Last Rows of the Dataset")
    st.dataframe(df1.tail())

    # Store the uploaded dataset in session_state for further processing
    st.session_state.df1 = df1

# Generate Technical Indicators
if 'df1' in st.session_state and st.sidebar.button("Generate Technical Indicators"):
    st.warning("Generating Technical Indicators... Please wait.")
    
    df1 = st.session_state.df1
    df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')

    # Calculate technical indicators (same as before)
    df1['SMA'] = df1['Close'].rolling(window=14).mean()
    df1['WMA'] = df1['Close'].rolling(window=14).apply(lambda x: np.dot(x, np.arange(1, len(x)+1))/np.sum(np.arange(1, len(x)+1)), raw=True)
    df1['Momentum'] = df1['Close'].diff(4)

    high_14 = df1['High'].rolling(window=14).max()
    low_14 = df1['Low'].rolling(window=14).min()
    df1['StochasticK'] = (df1['Close'] - low_14) / (high_14 - low_14) * 100
    df1['StochasticD'] = df1['StochasticK'].rolling(window=3).mean()

    delta = df1['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df1['RSI'] = 100 - (100 / (1 + rs))

    df1['MACD'] = df1['Close'].ewm(span=12, adjust=False).mean() - df1['Close'].ewm(span=26, adjust=False).mean()
    df1['MACD_signal'] = df1['MACD'].ewm(span=9, adjust=False).mean()

    df1['WilliamsR'] = (high_14 - df1['Close']) / (high_14 - low_14) * -100
    df1['A_D'] = df1['High'].diff(1) - df1['Low'].diff(1)
    df1['A_D'] = df1['A_D'].rolling(window=14).mean()

    tp = (df1['High'] + df1['Low'] + df1['Close']) / 3
    sma_tp = tp.rolling(window=14).mean()
    mean_deviation = (tp - sma_tp).abs().rolling(window=14).mean()
    df1['CCI'] = (tp - sma_tp) / (0.015 * mean_deviation)

    st.success("✅ Technical Indicators Generated!")

    # Clean NaN values
    st.warning("Cleaning NaN values from the dataset...")
    df1_cleaned = df1.dropna()

    st.session_state.df1_cleaned = df1_cleaned

    st.subheader("Cleaned Data with Technical Indicators")
    st.dataframe(df1_cleaned.head(50))

# Convert Close Price into Trend (Up/Down)
if 'df1_cleaned' in st.session_state:
    df1_cleaned = st.session_state.df1_cleaned
    df1_cleaned['Trend_Close'] = df1_cleaned['Close'].diff().apply(lambda x: 1 if x > 0 else -1)

    # Convert other technical indicators into trends (same as before)
    # Update df1_cleaned with Trend_SMA, Trend_WMA, etc.

# Run predictions for all models
if st.sidebar.button("Run Forecast"):
    st.success("Running Forecast for all models...")

    # Define the features for prediction
    features_discrete = df1_cleaned[['Trend_SMA', 'Trend_WMA', 'Trend_Momentum', 'Trend_StochasticK', 
                                    'Trend_StochasticD', 'Trend_RSI', 'Trend_MACD', 'Trend_WilliamsR', 
                                    'Trend_A_D', 'Trend_CCI']]  # feature columns

    # Make predictions for each model
    lgb_pred = lgb_model.predict(features_discrete)
    rf_pred = rf_model.predict(features_discrete)
    svm_pred = svm_model.predict(features_discrete)

    # Store the predictions in the dataframe
    df1_cleaned["LightGBM_Predicted"] = lgb_pred
    df1_cleaned["Random_Forest_Predicted"] = rf_pred
    df1_cleaned["SVM_Predicted"] = svm_pred

    st.subheader("Prediction Results (Actual vs Predicted Trend)")

    # Create a new dataframe showing only the actual trend and predicted trends
    df_predicted = df1_cleaned[['Date', 'Trend_Close', 'LightGBM_Predicted', 'Random_Forest_Predicted', 'SVM_Predicted']] 

    # Add correctness check directly to predicted columns with a gap between predicted value and tick
    df_predicted['LightGBM_Correct'] = np.where((df_predicted['Trend_Close'] == 1) & (df_predicted['LightGBM_Predicted'] == 1) | 
                                                (df_predicted['Trend_Close'] == -1) & (df_predicted['LightGBM_Predicted'] == -1), '✔️', '❌')
    
    df_predicted['Random_Forest_Correct'] = np.where((df_predicted['Trend_Close'] == 1) & (df_predicted['Random_Forest_Predicted'] == 1) | 
                                                      (df_predicted['Trend_Close'] == -1) & (df_predicted['Random_Forest_Predicted'] == -1), '✔️', '❌')
    
    df_predicted['SVM_Correct'] = np.where((df_predicted['Trend_Close'] == 1) & (df_predicted['SVM_Predicted'] == 1) | 
                                            (df_predicted['Trend_Close'] == -1) & (df_predicted['SVM_Predicted'] == -1), '✔️', '❌')

    # Metrics for LightGBM
    lgb_accuracy = accuracy_score(df_predicted['Trend_Close'], df_predicted['LightGBM_Predicted'])
    lgb_precision = precision_score(df_predicted['Trend_Close'], df_predicted['LightGBM_Predicted'], average='binary', pos_label=1)
    lgb_recall = recall_score(df_predicted['Trend_Close'], df_predicted['LightGBM_Predicted'], average='binary', pos_label=1)
    lgb_f1 = f1_score(df_predicted['Trend_Close'], df_predicted['LightGBM_Predicted'], average='binary', pos_label=1)
    lgb_confusion = confusion_matrix(df_predicted['Trend_Close'], df_predicted['LightGBM_Predicted'])

    # Metrics for Random Forest
    rf_accuracy = accuracy_score(df_predicted['Trend_Close'], df_predicted['Random_Forest_Predicted'])
    rf_precision = precision_score(df_predicted['Trend_Close'], df_predicted['Random_Forest_Predicted'], average='binary', pos_label=1)
    rf_recall = recall_score(df_predicted['Trend_Close'], df_predicted['Random_Forest_Predicted'], average='binary', pos_label=1)
    rf_f1 = f1_score(df_predicted['Trend_Close'], df_predicted['Random_Forest_Predicted'], average='binary', pos_label=1)
    rf_confusion = confusion_matrix(df_predicted['Trend_Close'], df_predicted['Random_Forest_Predicted'])

    # Metrics for SVM
    svm_accuracy = accuracy_score(df_predicted['Trend_Close'], df_predicted['SVM_Predicted'])
    svm_precision = precision_score(df_predicted['Trend_Close'], df_predicted['SVM_Predicted'], average='binary', pos_label=1)
    svm_recall = recall_score(df_predicted['Trend_Close'], df_predicted['SVM_Predicted'], average='binary', pos_label=1)
    svm_f1 = f1_score(df_predicted['Trend_Close'], df_predicted['SVM_Predicted'], average='binary', pos_label=1)
    svm_confusion = confusion_matrix(df_predicted['Trend_Close'], df_predicted['SVM_Predicted'])

    # Display performance metrics
    st.subheader("Performance Metrics")
    st.markdown(f"**LightGBM Metrics:**")
    st.markdown(f"Accuracy: {lgb_accuracy:.4f}")
    st.markdown(f"Precision: {lgb_precision:.4f}")
    st.markdown(f"Recall: {lgb_recall:.4f}")
    st.markdown(f"F1 Score: {lgb_f1:.4f}")
    st.markdown(f"Confusion Matrix:\n{lgb_confusion}")

    st.markdown(f"**Random Forest Metrics:**")
    st.markdown(f"Accuracy: {rf_accuracy:.4f}")
    st.markdown(f"Precision: {rf_precision:.4f}")
    st.markdown(f"Recall: {rf_recall:.4f}")
    st.markdown(f"F1 Score: {rf_f1:.4f}")
    st.markdown(f"Confusion Matrix:\n{rf_confusion}")

    st.markdown(f"**SVM Metrics:**")
    st.markdown(f"Accuracy: {svm_accuracy:.4f}")
    st.markdown(f"Precision: {svm_precision:.4f}")
    st.markdown(f"Recall: {svm_recall:.4f}")
    st.markdown(f"F1 Score: {svm_f1:.4f}")
    st.markdown(f"Confusion Matrix:\n{svm_confusion}")

    # Plot Confusion Matrix for each model
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # LightGBM Confusion Matrix
    sns.heatmap(lgb_confusion, annot=True, fmt="d", cmap="Blues", ax=ax[0], cbar=False)
    ax[0].set_title('LightGBM Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')

    # Random Forest Confusion Matrix
    sns.heatmap(rf_confusion, annot=True, fmt="d", cmap="Blues", ax=ax[1], cbar=False)
    ax[1].set_title('Random Forest Confusion Matrix')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')

    # SVM Confusion Matrix
    sns.heatmap(svm_confusion, annot=True, fmt="d", cmap="Blues", ax=ax[2], cbar=False)
    ax[2].set_title('SVM Confusion Matrix')
    ax[2].set_xlabel('Predicted')
    ax[2].set_ylabel('Actual')

    st.pyplot(fig)

