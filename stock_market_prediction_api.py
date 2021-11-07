import streamlit as st
from nsepy import get_history
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#Libraries for model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

#52 week high Price Calculation
@st.cache
def highprice(df):
    start_date = date.today() - timedelta(365)
    year_data = df["High"].loc[(df.index).date >= start_date]
    high_price = max(year_data)
    return high_price

#52 week Low Price Calculation
@st.cache
def lowprice(df):
    start_date = date.today() - timedelta(365)
    year_data = df["Low"].loc[(df.index).date >= start_date]
    low_price = min(year_data)
    return low_price

#Profit/Loss percentage
@st.cache
def pctchange(df, n):
    data = df[["Close"]].iloc[-n:]
    data["Profit/Loss %"] = (data.pct_change())*100
    data.index = (data.index).date
    profit_data = data.dropna()
    profit_data.rename(columns={"Close" : "Closing Price"}, inplace=True)
    return profit_data

def color_negative_red(val):
    color = 'red' if val < 0 else 'green'
    return 'color: %s' % color
#-----------------------------------------------------------------------------------------
#Downlaod NSE data for specific company
@st.cache
def getnsedata(comp_name):
    data = get_history(symbol=comp_name,
                       start=date(2016, 1, 1),
                       end=date.today())
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
    return data
#-----------------------------------------------------------------------------------------
#Data Preparation for LSTM model
@st.cache
def datapreparation(data_scaled):
    
    time_step = 30
    dataX, dataY = [], []
    
    # convert an array of values into a dataset matrix using time step
    for i in range(len(data_scaled)-time_step):
        dataX.append(data_scaled[i:(i+time_step), 0])    
        dataY.append(data_scaled[i + time_step, 0])
        
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    x_train = dataX.reshape(dataX.shape[0], dataX.shape[1] , 1)
    y_train = dataY
    
    return x_train, y_train
#-----------------------------------------------------------------------------------------
#LSTM Model Building
def model_building(neuron1, dropout_rate, x_train, y_train, epochs, batch_size):
    from numpy.random import seed
    seed(1)
    import tensorflow
    tensorflow.random.set_seed(2)

    model_lstm=Sequential()
    model_lstm.add(LSTM(neuron1,return_sequences=True,input_shape=(x_train.shape[1], 1)))
    model_lstm.add(Dropout(dropout_rate))
    model_lstm.add(LSTM(neuron1,return_sequences=True))
    model_lstm.add(Dropout(dropout_rate))
    model_lstm.add(LSTM(neuron1,return_sequences=True))
    model_lstm.add(Dropout(dropout_rate))
    model_lstm.add(LSTM(neuron1))
    model_lstm.add(Dropout(dropout_rate))

    model_lstm.add(Dense(1, activation='linear'))
    model_lstm.compile(loss='mean_squared_error',optimizer='adam')
    
    model_lstm.fit(x_train,y_train,epochs=epochs, batch_size=batch_size, verbose=0)
    return model_lstm

#-----------------------------------------------------------------------------------------
#Prediction using LSTM model
def prediction(trained_model, data_scaled, scalar, no_of_days_of_prediction): #prediction_till_date parameter should be added
    x_data = data_scaled.copy()
    predicted_price_list = []
    time_steps = 30
    
    for _ in range(no_of_days_of_prediction):
        x_data = x_data[-time_steps:]
        x_data = x_data.reshape(1, time_steps, 1)
        predicted_price = trained_model.predict(x_data)[0][0]
        predicted_price_list = np.append(predicted_price_list, predicted_price)
        x_data = np.append(x_data, predicted_price)
        
    forecasted_prices_list = scalar.inverse_transform((np.array(predicted_price_list)).reshape(-1,1))
    predicted_value_df = pd.DataFrame(forecasted_prices_list, index=datarange(no_of_days_of_prediction), 
                                      columns=["Predicted Closing Price"])
    predicted_value_df.index = pd.to_datetime(predicted_value_df.index, format="%Y-%m-%d")
    predicted_value_df.index = (predicted_value_df.index).date                                                 
    return predicted_value_df


def datarange(no_of_days_of_prediction):
    datelist = []
    for i in range (1,no_of_days_of_prediction+1):
        a = (date.today() +timedelta(days=i))
        datelist.append(a.strftime("%Y-%m-%d"))
    return datelist
#-----------------------------------------------------------------------------------------
#Above code deficts the user defined functions
#-----------------------------------------------------------------------------------------


#Page setup
st.set_page_config(
     page_title="Stock Market Prediction App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",)

st.title('''Stock market Predictions''')
#Imgae display
st.image("image.jpg")

page_name=['NSE','BSE']
page=st.sidebar.radio("Security Exchanges", page_name)

NSE_Stocks = {"Select Stock":"","State Bank of India":"SBIN", "Infosys Limited":"INFY", "Avenue Supermarts Ltd (DMart)":"DMART"}
BSE_Stocks = {"Select Stock":"","State Bank of India":"SBIN", "Infosys Limited":"INFY", "Avenue Supermarts Ltd (DMart)":"DMART"}

best_parameters = {"SBIN":{"neuron1":40,"dropout_rate":0.1,"epochs":125,"batch_size":30},
                  "INFY":{"neuron1":40,"dropout_rate":0.1,"epochs":80,"batch_size":40},
                  "DMART":{"neuron1":60,"dropout_rate":0.0,"epochs":150,"batch_size":60}}

if page == 'NSE': 
    stocks = ('Select stock','SBIN', 'INFY', 'DMART')
    selected_stock = st.selectbox('Listed Stocks', NSE_Stocks.keys())
     
    
else:
    stocks = ('Select stock','SBIN', 'INFY', 'DMART')
    selected_stock = st.selectbox('Listed Stocks', BSE_Stocks.keys())

st.write(f"## {selected_stock}")
st.write("___________________________________________________________")

if NSE_Stocks[selected_stock] != "":
    data_df = getnsedata(NSE_Stocks[selected_stock])
    latest_close_price = data_df.iloc[-1,7]
    
    col1, col2, col3 = st.columns([1.5,1,1])
    col1.write(f"### Closing Price ({((data_df.index[-1]).date())})")
    col1.write(latest_close_price)
    col2.write("### 52 Week High")
    col2.write(highprice(data_df))
    col3.write("### 52 Week Low")
    col3.write(lowprice(data_df))
    
    st.write("___________________________________________________________")
    st.line_chart(data_df["Close"])
    st.write("___________________________________________________________")
    st.subheader("Profit and Loss Summary of Last 5 Days")
    profit_data = (pctchange(data_df, 6)).style.applymap(color_negative_red)
    col1, col2, col3 = st.columns([2,3,2])
    col2.write(profit_data)
    
    #normalisation of data
    scaler=MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(pd.DataFrame(data_df.iloc[:,7]))
    
    x_train, y_train = datapreparation(data_scaled)

    neuron1, dropout_rate,epochs, batch_size = best_parameters[NSE_Stocks[selected_stock]].values()
    
    model_lstm = model_building(neuron1, dropout_rate, x_train, y_train, epochs, batch_size)
    
    predicted_value = prediction(model_lstm, data_scaled, scaler, 10)
    st.write("___________________________________________________________")
    st.subheader("Predictions")
    st.dataframe(predicted_value)
