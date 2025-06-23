import pandas as pd
import numpy as np
import streamlit as st
import joblib  # For loading the pre-trained models
import plotly.graph_objects as go

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

# Load the pre-trained models
def load_models():
    lgb_model_path = "best_lgb_discrete.pkl"
    rf_model_path = "best_rf_discrete.pkl"
    svm_model_path = "best_svm_discrete.pkl"

    lgb_model = joblib.load(lgb_model_path)
    rf_model = joblib.load(rf_model_path)
    svm_model = joblib.load(svm_model_path)
    
    return lgb_model, rf_model, svm_model

lgb_model, rf_model, svm_model = load_models()

# User Guide Section with collapsible expander
def display_user_guide():
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
        """)

# File Upload
def upload_file():
    uploaded_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv"])
    if uploaded_file:
        df1 = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.subheader("First Rows of the Dataset")
        st.dataframe(df1.head())
        st.subheader("Last Rows of the Dataset")
        st.dataframe(df1.tail())

        # Store the uploaded dataset in session_state for further processing
        st.session_state.df1 = df1
    return df1 if uploaded_file else None

# Generate Technical Indicators
def generate_technical_indicators(df):
    st.warning("Generating Technical Indicators... Please wait.")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Calculate technical indicators
    df['SMA'] = df['Close'].rolling(window=14).mean()
    df['WMA'] = df['Close'].rolling(window=14).apply(lambda x: np.dot(x, np.arange(1, len(x)+1))/np.sum(np.arange(1, len(x)+1)), raw=True)
    df['Momentum'] = df['Close'].diff(4)

    high_14 = df['High'].rolling(window=14).max()
    low_14 = df['Low'].rolling(window=14).min()
    df['StochasticK'] = (df['Close'] - low_14) / (high_14 - low_14) * 100
    df['StochasticD'] = df['StochasticK'].rolling(window=3).mean()

    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['WilliamsR'] = (high_14 - df['Close']) / (high_14 - low_14) * -100
    df['A_D'] = df['High'].diff(1) - df['Low'].diff(1)
    df['A_D'] = df['A_D'].rolling(window=14).mean()

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=14).mean()
    mean_deviation = (tp - sma_tp).abs().rolling(window=14).mean()
    df['CCI'] = (tp - sma_tp) / (0.015 * mean_deviation)

    st.success("✅ Technical Indicators Generated!")

    # Clean NaN values
    st.warning("Cleaning NaN values from the dataset...")
    df_cleaned = df.dropna()

    st.session_state.df_cleaned = df_cleaned
    st.subheader("Cleaned Data with Technical Indicators")
    st.dataframe(df_cleaned.head(50))

# Convert Technical Indicators into Trends (using the approach you specified)
def convert_to_trends(df):
    df['Trend'] = np.where(df['Close'] > df['Close'].shift(1), 1, 
                           np.where(df['Close'] < df['Close'].shift(1), -1, 0))

    df['Trend_SMA'] = np.where(df['Close'] > df['SMA'], 1, 
                               np.where(df['Close'] <= df['SMA'], -1, 0))

    df['Trend_WMA'] = np.where(df['Close'] > df['WMA'], 1, 
                               np.where(df['Close'] <= df['WMA'], -1, 0))

    df['Trend_Momentum'] = np.where(df['Momentum'] > 0, 1, 
                                    np.where(df['Momentum'] <= 0, -1, 0))

    df['Trend_StochasticK'] = np.where(df['StochasticK'] > df['StochasticK'].shift(1), 1, 
                                       np.where(df['StochasticK'] <= df['StochasticK'].shift(1), -1, 0))

    df['Trend_StochasticD'] = np.where(df['StochasticD'] > df['StochasticD'].shift(1), 1, 
                                       np.where(df['StochasticD'] <= df['StochasticD'].shift(1), -1, 0))

    df['Trend_RSI'] = np.where(df['RSI'] < 30, 1,
                                np.where(df['RSI'] > 70, -1,
                                np.where(df['RSI'] > df['RSI'].shift(1), 1,
                                np.where(df['RSI'] <= df['RSI'].shift(1), -1, 0))))

    df['Trend_MACD'] = np.where(df['MACD'] > df['MACD'].shift(1), 1, 
                                np.where(df['MACD'] <= df['MACD'].shift(1), -1, 0))

    df['Trend_WilliamsR'] = np.where(df['WilliamsR'] > df['WilliamsR'].shift(1), 1, 
                                     np.where(df['WilliamsR'] <= df['WilliamsR'].shift(1), -1, 0))

    df['Trend_A_D'] = np.where(df['A_D'] > df['A_D'].shift(1), 1, 
                               np.where(df['A_D'] <= df['A_D'].shift(1), -1, 0))

    df['Trend_CCI'] = np.where(df['CCI'] > df['CCI'].shift(1), 1, 
                               np.where(df['CCI'] <= df['CCI'].shift(1), -1, 0))

    st.subheader("Data with Trends")
    st.dataframe(df.head(100))  # Display the first 100 rows

    return df

# Prediction
def run_forecast(df, models):
    features_discrete = df[['Trend_SMA', 'Trend_WMA', 'Trend_Momentum', 'Trend_StochasticK', 
                            'Trend_StochasticD', 'Trend_RSI', 'Trend_MACD', 'Trend_WilliamsR', 
                            'Trend_A_D', 'Trend_CCI']]

    lgb_pred = models[0].predict(features_discrete)
    rf_pred = models[1].predict(features_discrete)
    svm_pred = models[2].predict(features_discrete)

    df["LightGBM_Predicted"] = lgb_pred
    df["Random_Forest_Predicted"] = rf_pred
    df["SVM_Predicted"] = svm_pred

    # Store predicted data in session_state to maintain state across app reruns
    st.session_state.df_predicted = df
    return df

# Visualization
def plot_predictions(df):
    # Access the data stored in session_state to ensure the predictions are available
    if 'df_predicted' not in st.session_state:
        st.error("Predictions not found. Please run the forecast first.")
        return

    df = st.session_state.df_predicted

    # Check if the necessary columns exist in the DataFrame
    required_columns = ['Trend_Close', 'LightGBM_Predicted', 'Random_Forest_Predicted', 'SVM_Predicted']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns for prediction visualization: {', '.join(missing_columns)}")
        return

    df_predicted = df[['Date', 'Trend_Close', 'LightGBM_Predicted', 'Random_Forest_Predicted', 'SVM_Predicted']]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Trend_Close'],
        mode='markers',
        name='Actual Price',
        marker=dict(color='black', size=8, symbol='circle')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['LightGBM_Predicted'],
        mode='markers',
        name='LightGBM Predicted',
        marker=dict(color='blue', size=6, symbol='diamond')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Random_Forest_Predicted'],
        mode='markers',
        name='Random Forest Predicted',
        marker=dict(color='red', size=6, symbol='square')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['SVM_Predicted'],
        mode='markers',
        name='SVM Predicted',
        marker=dict(color='green', size=6, symbol='triangle-up')
    ))

    fig.update_layout(
        title='📊 Actual vs Predicted XAU/USD Price Movements',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Model',
        hovermode='x unified',
        template='plotly_white',
        height=1000,
        width=1500
    )

    st.plotly_chart(fig)

# Main Code Flow
def main():
    display_user_guide()

    uploaded_file = upload_file()
    
    if uploaded_file is not None:
        df1 = uploaded_file
        if st.sidebar.button("Generate Technical Indicators"):
            generate_technical_indicators(df1)

        if 'df_cleaned' in st.session_state:
            df_cleaned = st.session_state.df_cleaned
            convert_to_trends(df_cleaned)

        if st.sidebar.button("Run Forecast"):
            st.success("Running Forecast for all models...")
            df_with_predictions = run_forecast(df_cleaned, [lgb_model, rf_model, svm_model])
            plot_predictions(df_with_predictions)

if __name__ == "__main__":
    main()
