import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy import interpolate
from sklearn.impute import KNNImputer

#from google.colab import drive
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.linear_model import BayesianRidge, Ridge
import math

sensor_data = pd.DataFrame()

@st.cache_data
def view_boxplot(dataframe_used):
  fig = plt.figure(figsize=(25, 15))
  sns.boxplot(data = dataframe_used)
  st.pyplot(fig)


#Defining IQR method to identify outliers 
@st.cache_resource
def find_outliers_IQR(df):
   Q1 = df.quantile(0.25)
   Q3 = df.quantile(0.75)
   IQR = Q3 - Q1
   outliers = df[((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR)))]
   return outliers

#Imputing Outliers Identified using IQR with NaN
@st.cache_data
def impute_outliers_IQR(df):
   Q1 = df.quantile(0.25)
   Q3 = df.quantile(0.75)
   IQR = Q3 - Q1
   upper = df[~(df>(Q3+1.5*IQR))].max()
   lower = df[~(df<(Q1-1.5*IQR))].min()
   df = np.where(df > upper, df.replace('?', np.NaN, inplace = True),
          np.where(df < lower, df.replace('?', np.NaN, inplace = True),
           df))

   return df


st.title("Data Purifier")

uploaded_file = st.file_uploader("Upload the Sensor Data [Currently accepting Excel(.xlsx) Format]")

if uploaded_file is not None:
  with st.spinner('Loading...'):
    status_placeholder = st.empty()
    status_placeholder.text("File uploading...")
    sensor_data = pd.read_excel(uploaded_file)
    status_placeholder.text("File uploaded successfully.")
    bx = st.button('Visualize Ouliers using Boxplot')
    if bx:
      status_placeholder = st.empty()
      status_placeholder.text("Creating plot...")
      view_boxplot(sensor_data.iloc[:,1:])
      status_placeholder.text("Plot created successfully")

           
    y = st.text_input('Enter the sensor number for which you wish to perform imputations')
    x="s_"+y
       
      # Show the "Find Outliers" button
    if x in sensor_data.columns:
          outliers = find_outliers_IQR(sensor_data[x])
          st.write("**Please find more details about this sensor:**")
          st.write("Number of outliers in sensor: ", str(len(outliers)))
          st.write("Max outlier value: ", str(outliers.max()))
          st.write("Min outlier value: ", str(outliers.min()))

          #Final NaN imputed outlier data frame    
          sensor_data_NaN = pd.DataFrame()
          sensor_data_NaN[x] = impute_outliers_IQR(sensor_data[x])
          
          #Data frame for Box Jenkins & Exponential Smoothening Imputation
          sensor_data_add = pd.DataFrame()
          sensor_data_add['Index'] = sensor_data['Index']
          sensor_data_add[x] = sensor_data[x]

          sensor_data_MA = pd.DataFrame()
          sensor_data_MA['Index'] = sensor_data['Index']
          sensor_data_MA[x] = sensor_data[x]
          st.write(f"**Select your choice of visualization to assess Sensor {y}**")
          option = st.selectbox(
            'Visualize options:',
            ('Moving Range', 'Inter Quartile Range (IQR)'))
          # st.write('You selected:', option)
          with st.spinner('Loading...'):
            if(option=='Inter Quartile Range (IQR)'):
              st.write("IQR in processing....")
              fig1 = plt.figure(figsize=(10, 10))
              # sns.set(rc={"figure.figsize":(10, 10)})
              sns.boxplot(data = sensor_data[x])
              st.pyplot(fig1)
              # option1 = st.selectbox(
              #      'Imputation options:',
              #      ('MICE', 'KNN', 'Interpolation', 'Box Jenkins', 'Exponential Smoothening'),default=None)
              # st.write('You selected:', option1)
            elif(option=='Moving Range'):
                dfact = st.text_input('Enter a weight for moving range:')
                if not dfact:
                  st.warning("Default Value is 5")#st.write('Your entered:', dfact)
                else:
                  st.write("Moving Range graph in processing....")
                  sensor_data_MA_update = sensor_data_MA.filter(['Index',x], axis=1).copy() 
                  sensor_data_MA_update['diff'] = sensor_data_MA_update[[x]].diff(periods=1) 
                  sensor_data_MA_update['abs_diff'] = sensor_data_MA_update['diff'].abs() 
                  sensor_data_MA_update['cum_diff'] = ""
                  for i in range(0,3):
                      sensor_data_MA_update.cum_diff[i] = 0
                  for i in range(3, len(sensor_data_MA_update[x])):
                      sensor_data_MA_update['cum_diff'][i] = (((sensor_data_MA_update['Index'][i] - 2) * (sensor_data_MA_update['cum_diff'][i - 1])) + sensor_data_MA_update['abs_diff'][i])/(sensor_data_MA_update['Index'][i] - 1)
                  sensor_data_MA_update['sim_MA'] = sensor_data_MA_update['cum_diff']*int(dfact)
                  plt.plot(sensor_data_MA_update['sim_MA'], c = 'g', label = 'Simple Moving Range')
                  plt.legend()
                  st.pyplot(plt)  
          # option1 = st.selectbox(
          # 'Imputation options:',
          # ('MICE', 'KNN', 'Interpolation', 'Box Jenkins', 'Exponential Smoothening'),index=None)
          # st.write('You selected:', option1)
    else:
        st.warning(f"Enter a correct sensor number that exists in dataset")
 


    
    alpha_value = st.number_input("Enter an alpha value to be used by Imputation algoriths below")
    #st.write("You set alpha to:",alpha_value)  
    x_alpha = float(alpha_value)
    if x_alpha!=0:
        st.write('**Select an Imputation algorithm from the following options:**')
        option1 = st.selectbox(
        'Imputation options:',
        ('MICE', 'KNN', 'Interpolation', 'Box Jenkins', 'Exponential Smoothening'))
        st.write('You selected:', option1)
        if(option1=='MICE'):
          st.write("MICE Imputation in processing....")
          ##RMSE Function 
          def mape(actual, pred):
            return np.mean(np.abs((actual-pred)/actual)* 100)
          ##MICE Imputation Model
          sensor_mice = sensor_data_NaN.filter([x], axis=1).copy()
          mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=100, imputation_order='ascending')
          df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(sensor_mice), columns=sensor_mice.columns)
          ##Visualizing Original Vs Imputed Dataset
          fig, axs = plt.subplots(2)
          axs[0].plot(sensor_data[x], c = 'g')
          axs[0].set_title("Original Data")
          axs[0].set_ylim(0,35)
          axs[1].plot(df_mice_imputed, c = 'b')
          axs[1].set_title("MICE Imputation")
          axs[1].set_ylim(0,35)
          # using padding
          fig.tight_layout()
          # plt.legend()
          st.pyplot(fig)
          # sensor_data[x].plot(figsize=(20,10), ylim=(0,35), title = "Original Data", color = ['#088F8F'])
          # df_mice_imputed.plot(figsize=(20,10), ylim=(0,35), title = "MICE Imputation", color = ['#088F8F'])
          ##Determining RMSE Value 
          rmse_pandas = sqrt(mean_squared_error(sensor_data[x], df_mice_imputed))
          st.write('MICE Imputation RMSE :',rmse_pandas)
          ##Downloading Imputed Dataset
          sensor_data_update = pd.DataFrame()
          sensor_data_update = sensor_data.copy()
          sensor_data_update[x] = df_mice_imputed
          sensor_data_update.to_csv("Mice_Imputation.csv")
              
        elif(option1=='KNN'):
          st.write("KNN Imputation in processing....")
          ##RMSE Function
          def mape(actual, pred):
            return np.mean(np.abs((actual-pred)/actual)* 100)
          ##KNN Imputation Model 
          sensor_KNN = sensor_data_NaN.filter([x], axis=1).copy()
          knn_imputer = KNNImputer(n_neighbors=100)
          df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(sensor_KNN), columns=sensor_KNN.columns)

          ##Visualizing Original Vs Imputed Dataset
          fig, axs = plt.subplots(2)
          axs[0].plot(sensor_data[x], c = 'g')
          axs[0].set_title("Original Data")
          axs[0].set_ylim(0,35)
          axs[1].plot(df_knn_imputed, c = 'b')
          axs[1].set_title("KNN Imputation")
          axs[1].set_ylim(0,35)
          # using padding
          fig.tight_layout()
          # plt.legend()
          st.pyplot(fig)
          ##Visualizing Original Vs Imputed Dataset 
          # sensor_data[x].plot(figsize=(20,10), ylim=(0,35), title = "Original Data", color = ['#088F8F'])
          # df_knn_imputed.plot(figsize=(20,10), ylim=(0,35), title = "KNN Imputation", color = ['#088F8F'])
          ##Determining RMSE Value 
          rmse_pandas = sqrt(mean_squared_error(sensor_data[x], df_knn_imputed))
          st.write('KNN Imputation RMSE :',rmse_pandas)
          ##Downloading Imputed Dataset
          sensor_data_update = pd.DataFrame()
          sensor_data_update = sensor_data.copy()
          sensor_data_update[x] = df_knn_imputed
          sensor_data_update.to_csv("KNN_Imputation.csv")

        elif(option1=='Interpolation'):
          st.write("Interpolation Imputation in processing....")
          ##RMSE Function
          def mape(actual, pred):
            return np.mean(np.abs((actual-pred)/actual)* 100)
          ##Interpolation Imputation Model 
          
          sensor_interpolate = sensor_data_NaN.copy()
          df_inter = pd.DataFrame(sensor_interpolate[x].fillna(method='ffill'))

          ##Visualizing Original Vs Imputed Dataset
          fig, axs = plt.subplots(2)
          axs[0].plot(sensor_data[x], c = 'g')
          axs[0].set_title("Original Data")
          axs[0].set_ylim(0,35)
          axs[1].plot(df_inter, c = 'b')
          axs[1].set_title("Interpolation Imputation")
          axs[1].set_ylim(0,35)
          # using padding
          fig.tight_layout()
          # plt.legend()
          st.pyplot(fig)
          
          ##Visualizing Original Vs Imputed Dataset 
          # sensor_data[x].plot(figsize=(20,10), ylim=(0,35), title = "Original Data", color = ['#088F8F'])
          # df_inter.plot(figsize=(20,10), ylim=(0,35), title = "Interpolation Imputation", color = ['#088F8F'])
          ##Determining RMSE Value 
          rmse_pandas = sqrt(mean_squared_error(sensor_data[x], df_inter))
          st.write('Interpolation Imputation RMSE :',rmse_pandas)
          ##Downloading Imputed Dataset
          sensor_data_update = pd.DataFrame()
          sensor_data_update = sensor_data.copy()
          sensor_data_update[x] = df_inter
          sensor_data_update.to_csv("Interpolation_Imputation.csv")

        elif(option1=='Box Jenkins'):
          print("Box Jenkins Imputation in processing....")
          ##RMSE Function
          def mape(actual, pred):
            return np.mean(np.abs((actual-pred)/actual)* 100)
          ##Preparing Data for Box Jenkins - Chained assignments
          sensor_data_NaN_update = sensor_data_add.filter(['Index',x], axis=1).copy() 
          sensor_data_NaN_update['diff'] = sensor_data_NaN_update[[x]].diff(periods=1) 
          sensor_data_NaN_update['abs_diff'] = sensor_data_NaN_update['diff'].abs() 
          sensor_data_NaN_update['cum_diff'] = ""
          for i in range(0,3):
            sensor_data_NaN_update.cum_diff[i] = 0
          for i in range(3, len(sensor_data_NaN_update[x])):
            sensor_data_NaN_update['cum_diff'][i] = (((sensor_data_NaN_update['Index'][i] - 2) * (sensor_data_NaN_update['cum_diff'][i - 1])) + sensor_data_NaN_update['abs_diff'][i])/(sensor_data_NaN_update['Index'][i] - 1)
            sensor_data_NaN_update['sim_MA'] = sensor_data_NaN_update['cum_diff']*int(dfact)
          sensor_data_NaN_update['outlier'] = ""
          for i in range(1, len(sensor_data_NaN_update[x])):
            if(sensor_data_NaN_update[x][i] == 0 or (sensor_data_NaN_update['abs_diff'][i] > sensor_data_NaN_update['sim_MA'][i])):
              sensor_data_NaN_update['outlier'][i] = 1
            else:
              sensor_data_NaN_update['outlier'][i] = 0
          ##Box Jenkins Imputation Model 
          forecast_box_jen = [sensor_data_NaN_update[x][0]]
          for i in range(1, len(sensor_data_NaN_update[x])):
            if(sensor_data_NaN_update.outlier[i] == 1):
              predict = round(forecast_box_jen[i - 1] + (x_alpha) * (forecast_box_jen[i - 1] - forecast_box_jen[i - 2]),2)
              forecast_box_jen.append(predict)
            else:
              predict = sensor_data_NaN_update[x][i]
              forecast_box_jen.append(predict)
          ##Visualizing Original Vs Imputed Dataset
          df_forecast_box_jen = pd.DataFrame(forecast_box_jen)

          ##Visualizing Original Vs Imputed Dataset
          fig, axs = plt.subplots(2)
          axs[0].plot(sensor_data[x], c = 'g')
          axs[0].set_title("Original Data")
          axs[0].set_ylim(0,35)
          axs[1].plot(df_forecast_box_jen, c = 'b')
          axs[1].set_title("Box Jenkins Imputed")
          axs[1].set_ylim(0,35)
          # using padding
          fig.tight_layout()
          # plt.legend()
          st.pyplot(fig)
          # sensor_data[x].plot(figsize=(20,10), ylim=(0,35), title = "Original", color = ['#088F8F'])
          # df_forecast_box_jen.plot(figsize=(20, 10), ylim=(0,35), title = "Box Jenkins Imputed", color = ['#088F8F'])
          rmse_pandas = sqrt(mean_squared_error(sensor_data[x], df_forecast_box_jen))
          st.write('Box Jenkins Imputation RMSE :',rmse_pandas)
          ##Downloading Imputed Dataset
          sensor_data_update = pd.DataFrame()
          sensor_data_update = sensor_data.copy()
          sensor_data_update[x] = forecast_box_jen
          sensor_data_update.to_csv("Box_Jenkins_Imputation.csv")

        elif(option1=='Exponential Smoothening'):
          st.write("Exponential Smoothening Imputation in processing....")
          ##RMSE Function
          def mape(actual, pred):
            return np.mean(np.abs((actual-pred)/actual)* 100)
          ##Preparing Data for Exponential Smoothening - Chained assignments 
          sensor_data_NaN_update = sensor_data_add.filter(['Index',x], axis=1).copy() 
          sensor_data_NaN_update['diff'] = sensor_data_NaN_update[[x]].diff(periods=1) 
          sensor_data_NaN_update['abs_diff'] = sensor_data_NaN_update['diff'].abs() 
          sensor_data_NaN_update['cum_diff'] = ""
          for i in range(0,3):
            sensor_data_NaN_update.cum_diff[i] = 0
          for i in range(3, len(sensor_data_NaN_update[x])):
            sensor_data_NaN_update['cum_diff'][i] = ((((sensor_data_NaN_update['Index'][i] - 2) * (sensor_data_NaN_update['cum_diff'][i - 1])) + sensor_data_NaN_update['abs_diff'][i])/(sensor_data_NaN_update['Index'][i] - 1))
            sensor_data_NaN_update['sim_MA'] = sensor_data_NaN_update['cum_diff']*int(dfact)
          sensor_data_NaN_update['outlier'] = ""
          for i in range(1, len(sensor_data_NaN_update[x])):
            if(sensor_data_NaN_update['abs_diff'][i] > sensor_data_NaN_update['sim_MA'][i] ):
              sensor_data_NaN_update['outlier'][i] = 1
            else:
              sensor_data_NaN_update['outlier'][i] = 0
          ##Exponential Smoothening Imputation Model         
          forecast_exp_smooth = [sensor_data_NaN_update[x][0]]
          for i in range(1, len(sensor_data_NaN_update[x])):
            if(sensor_data_NaN_update.outlier[i] == 1):
              predict = round(x_alpha * sensor_data_NaN_update[x][i - 1] + (1 - x_alpha) * forecast_exp_smooth[i - 1],2)
              forecast_exp_smooth.append(predict)
            else:
              predict = sensor_data_NaN_update[x][i]
              forecast_exp_smooth.append(predict)
      
          ##Visualizing Original Vs Imputed Dataset
          df_forecast_exp_smooth = pd.DataFrame(forecast_exp_smooth)

          ##Visualizing Original Vs Imputed Dataset
          fig, axs = plt.subplots(2)
          axs[0].plot(sensor_data[x], c = 'g')
          axs[0].set_title("Original Data")
          axs[0].set_ylim(0,35)
          axs[1].plot(df_forecast_exp_smooth, c = 'b')
          axs[1].set_title("Exponential Smoothening Imputed")
          axs[1].set_ylim(0,35)
          # using padding
          fig.tight_layout()
          # sensor_data[x].plot(figsize=(20,10), title = "Original", color = ['#088f8f'])
          # df_forecast_exp_smooth.plot(figsize=(20, 10), title = "Exponential Smoothening Imputed", color = ['#088F8F'])
          ##Determining RMSE Value 
          rmse_pandas = sqrt(mean_squared_error(sensor_data[x], df_forecast_exp_smooth))
          st.write('Exponential Smoothening Imputation RMSE :',rmse_pandas)
          ##Downloading Imputed Dataset
          sensor_data_update = pd.DataFrame()
          sensor_data_update = sensor_data.copy()
          sensor_data_update[x] = forecast_exp_smooth
          sensor_data_update.to_csv("Exp_Smooth_Imputation.csv")
        else:
          st.write('Select valid option from dropdown')
    else:
      st.warning("Default value for alpha is 0.5")

# else:
#    st.warning("Please upload a file")
# # view_boxplot(df.iloc[:,1:])
# # plt.show()
