import streamlit as st
import pandas as pd
import numpy as np
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

    # Calculate technical indicators
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

    # Create a CSV for download
    def create_download_link(df, filename):
        csv = df.to_csv(index=False)
        
        # Saving the predictions in session state so they persist after the download
        if 'downloaded_data' not in st.session_state:
            st.session_state.downloaded_data = df
            
        return st.download_button(
            label=f"Download {filename}",
            data=csv,
            file_name=filename,
            mime='text/csv'
        )
    
    # Create the download link for technical indicators CSV
    create_download_link(df1_cleaned, "technical_indicators.csv")
    
    # Add this to re-display the prediction data after the download is triggered
    if 'downloaded_data' in st.session_state:
        st.subheader("Technical Indicator (After Download)")
        st.dataframe(st.session_state.downloaded_data)  # Display the downloaded data again
        

# Convert Close Price into Trend (Up/Down)
if 'df1_cleaned' in st.session_state:
    df1_cleaned = st.session_state.df1_cleaned
    df1_cleaned['Trend_Close'] = df1_cleaned['Close'].diff().apply(lambda x: 1 if x > 0 else -1)

    # Convert other technical indicators into trends
    df1_cleaned['Trend_SMA'] = df1_cleaned['SMA'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_WMA'] = df1_cleaned['WMA'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_Momentum'] = df1_cleaned['Momentum'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_StochasticK'] = df1_cleaned['StochasticK'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_StochasticD'] = df1_cleaned['StochasticD'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_RSI'] = df1_cleaned['RSI'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_MACD'] = df1_cleaned['MACD'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_WilliamsR'] = df1_cleaned['WilliamsR'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_A_D'] = df1_cleaned['A_D'].diff().apply(lambda x: 1 if x > 0 else -1)
    df1_cleaned['Trend_CCI'] = df1_cleaned['CCI'].diff().apply(lambda x: 1 if x > 0 else -1)

    df_trend = df1_cleaned[['Date', 'Trend_Close', 
                            'Trend_SMA', 'Trend_WMA', 'Trend_Momentum', 
                            'Trend_StochasticK', 'Trend_StochasticD', 'Trend_RSI', 
                            'Trend_MACD', 'Trend_WilliamsR', 'Trend_A_D', 'Trend_CCI']]

    # Display the trend data table
    st.subheader("Data with Trends")
    st.dataframe(df_trend.head(100))  # Display top 100 rows with trends only

    # Create a CSV for download
    def create_download_link(df, filename):
        csv = df.to_csv(index=False)
        
        # Saving the predictions in session state so they persist after the download
        st.session_state.downloaded_data = df
        
        return st.download_button(
            label=f"Download {filename}",
            data=csv,
            file_name=filename,
            mime='text/csv'
        )
    create_download_link(df_trend, "trend_data.csv")

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

    # Create separate columns for predicted values and ticks
    df_predicted['LightGBM_Predicted_Value'] = df_predicted['LightGBM_Predicted']
    df_predicted['Random_Forest_Predicted_Value'] = df_predicted['Random_Forest_Predicted']
    df_predicted['SVM_Predicted_Value'] = df_predicted['SVM_Predicted']

    # Now select the relevant columns for display (predicted value and correctness tick side by side)
    df_predicted = df_predicted[['Date', 'Trend_Close', 
                                 'LightGBM_Predicted_Value', 'LightGBM_Correct', 
                                 'Random_Forest_Predicted_Value', 'Random_Forest_Correct', 
                                 'SVM_Predicted_Value', 'SVM_Correct']]

    # Update the column headers to combine the "Predicted" and "Correct" into one header
    df_predicted.columns = ['Date', 'Actual Trend', 
                            'LightGBM (Predicted)', 'LightGBM (Correct)', 
                            'Random Forest (Predicted)', 'Random Forest (Correct)', 
                            'SVM (Predicted)', 'SVM (Correct)']

    # Display the filtered dataframe with actual and predicted trends side by side with correctness
    st.dataframe(df_predicted.head(100))  # Display top 10 predictions

    # Create downloadable links for prediction, technical indicators, and trends
    # Create a CSV for download
    def create_download_link(df, filename):
        csv = df.to_csv(index=False)
        
        # Saving the predictions in session state so they persist after the download
        st.session_state.downloaded_data = df
        
        return st.download_button(
            label=f"Download {filename}",
            data=csv,
            file_name=filename,
            mime='text/csv'
        )
        
    create_download_link(df_predicted, "prediction_results.csv")

    # Plot Actual vs Predicted (All Models)
    st.subheader(" Actual vs Predicted Prices (All Models)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df1_cleaned['Date'],
        y=df1_cleaned['Trend_Close'],
        mode='markers',
        name='Actual Price',
        marker=dict(color='black', size=8, symbol='circle')
    ))

    fig.add_trace(go.Scatter(
        x=df1_cleaned['Date'],
        y=df1_cleaned['LightGBM_Predicted'],
        mode='markers',
        name='LightGBM Predicted',
        marker=dict(color='blue', size=6, symbol='diamond')
    ))

    fig.add_trace(go.Scatter(
        x=df1_cleaned['Date'],
        y=df1_cleaned['Random_Forest_Predicted'],
        mode='markers',
        name='Random Forest Predicted',
        marker=dict(color='red', size=6, symbol='square')
    ))

    fig.add_trace(go.Scatter(
        x=df1_cleaned['Date'],
        y=df1_cleaned['SVM_Predicted'],
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
            template='plotly_white',  # This sets the plot background to white
            height=1000,
            width=1500,
            plot_bgcolor='white',  # Ensures the background of the plot itself is white
            paper_bgcolor='white',  # Ensures the area surrounding the plot is also white
            title_font=dict(
                family="Arial, sans-serif",  # Set font family for title
                size=40,  # Set font size for title
                color="black"  # Set title text color to black
            ),
            xaxis=dict(
                title='Date',
                title_font=dict(
                    family="Arial, sans-serif",  # Set font family for x-axis title
                    size=14,  # Set font size for x-axis title
                    color="black"  # Set x-axis title text color to black
                ),
                tickfont=dict(
                    family="Arial, sans-serif",  # Set font family for x-axis ticks
                    size=12,  # Set font size for x-axis ticks
                    color="black"  # Set x-axis tick text color to black
                ),
                rangeselector=dict(
                    buttons=list([

                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),  # Added 1 year button (backward)
                        dict(step="all")
                    ])


                ),

                rangeslider=dict(visible=True),
                type="date",
                tickangle=45,  # Optional: Rotates the ticks for better visibility
            ),
            yaxis=dict(
                title='Price',
                title_font=dict(
                    family="Arial, sans-serif",  # Set font family for y-axis title
                    size=14,  # Set font size for y-axis title
                    color="black"  # Set y-axis title text color to black
                ),
                tickfont=dict(
                    family="Arial, sans-serif",  # Set font family for y-axis ticks
                    size=12,  # Set font size for y-axis ticks
                    color="black"  # Set y-axis tick text color to black
                )
            ),
            legend_title_font=dict(
                family="Arial, sans-serif",  # Set font family for legend title
                size=14,  # Set font size for legend title
                color="black"  # Set legend title text color to black
            ),
            legend_font=dict(
                family="Arial, sans-serif",  # Set font family for legend items
                size=12,  # Set font size for legend items
                color="black"  # Set legend text color to black
            )
        )

    st.plotly_chart(fig)
