import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64
from PIL import Image

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


#st.title("DATA CLENZ")
image = Image.open ("DATACLENZ.png")
st.image(image)
st.write("**Welcome to DATA CLENZ application by Internet Sciences.**")
st.write("<u>Please follow the instructions:</u>",unsafe_allow_html=True)
st.write("1. This application accepts IOT Sensor data in .xlsx format")
st.write("2. User can visualize outliers from all sensors and can select one sensor for performing Imputation.")
st.write("3. User has the choice to select among 5 Imputation algorithms provided by this application and download clean imputed Sensor Data file.")
st.write("\n")
uploaded_file = st.file_uploader("**Upload the Sensor Data [Currently accepting Excel(.xlsx) Format]**")

def new_func():
    by = st.download_button()

if uploaded_file is not None:
  with st.spinner('Loading...'):
    status_placeholder = st.empty()
    status_placeholder.text("File uploading...")
    sensor_data_unrestricted = pd.read_excel(uploaded_file)
    status_placeholder.text("File uploaded successfully.")
    num_rows, num_cols = sensor_data_unrestricted.shape
    st.write(f'Number of rows: {num_rows}')
    st.write(f'Number of columns: {num_cols}')
    a = st.text_input('The Application will consider only first 14000 records. Enter Y to continue and N to abort')
    if a=='Y':
        sensor_data = sensor_data_unrestricted.iloc[:14000]

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
              dfact=5 # Setting default dfact to avoid errors in Imputation part
              outliers = find_outliers_IQR(sensor_data[x])
              st.write("**Please find more details about this sensor:**")
              st.write("Number of outliers in sensor: ", str(len(outliers)))
              st.write("Max outlier value: ", str(outliers.max()))
              st.write("Min outlier value: ", str(outliers.min()))
              outlier_df = pd.DataFrame(outliers)

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
 


        check=st.text_input("Please Enter Y to proceed with Imputations, N to abort")
        #alpha_value = st.number_input("Enter an alpha value to be used by Imputation algoriths below")
        #st.write("You set alpha to:",alpha_value)  
        #x_alpha = float(alpha_value)
        if check=="Y":
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
              b = st.text_input('Please enter Y if you wish to download output files ')
              if b == 'Y':
                sensor_data_update = pd.DataFrame()
                sensor_data_update = sensor_data.copy()
                sensor_data_update[x] = df_mice_imputed
                csv = sensor_data_update.to_csv(index = False)
                csv = sensor_data_update.to_csv(index = False)
                csv1 = outlier_df.to_csv(index = True)
                b64 = base64.b64encode(csv.encode()).decode()
                b641 = base64.b64encode(csv1.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="Mice_Imputation.csv">Download imputed file</a>'
                st.download_button(label='Download imputed file', data=csv, file_name='Mice_Imputation.csv', mime='text/csv')
                href1 = f'<a href="data:file/csv;base64,{b641}" download="Outlier.csv">Download outliers corrected in this imputation</a>'
                st.download_button(label='Download outliers corrected in this imputation', data=csv1, file_name='Outlier.csv', mime='text/csv')
              else:
                  st.write("Thank you for using the application")
              #"Mice_Imputation.csv"
             
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
              c = st.text_input('Please enter Y if you wish to download output files ')
              if c == 'Y':
                sensor_data_update = pd.DataFrame()
                sensor_data_update = sensor_data.copy()
                sensor_data_update[x] = df_knn_imputed
                csv = sensor_data_update.to_csv(index = False)
                csv1 = outlier_df.to_csv(index = True)
                b64 = base64.b64encode(csv.encode()).decode()
                b641 = base64.b64encode(csv1.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="KNN_Imputation.csv">Download</a>'
                st.download_button(label='Download', data=csv, file_name='KNN_Imputation.csv', mime='text/csv')
                href1 = f'<a href="data:file/csv;base64,{b641}" download="Outlier.csv">Download outliers corrected in this imputation</a>'
                st.download_button(label='Download outliers corrected in this imputation', data=csv1, file_name='Outlier.csv', mime='text/csv')
              else:
                  st.warning("Thank you for using the application")

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
              d = st.text_input('Please enter Y if you wish to download output files ')
              if d == 'Y':
                sensor_data_update = pd.DataFrame()
                sensor_data_update = sensor_data.copy()
                sensor_data_update[x] = df_inter
                csv = sensor_data_update.to_csv(index = False)
                csv1 = outlier_df.to_csv(index = True)
                b64 = base64.b64encode(csv.encode()).decode()
                b641 = base64.b64encode(csv1.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="Interpolation_Imputation.csv">Download</a>'
                st.download_button(label='Download', data=csv, file_name='Interpolation_Imputation.csv', mime='text/csv')
                href1 = f'<a href="data:file/csv;base64,{b641}" download="Outlier.csv">Download outliers corrected in this imputation</a>'
                st.download_button(label='Download outliers corrected in this imputation', data=csv1, file_name='Outlier.csv', mime='text/csv')
              else:
                  st.write("Thank you for using the application")

            elif(option1=='Box Jenkins'):
              alpha_value = st.number_input("Enter an alpha value to be used by Box Jenkins algorithm:")
              if not dfact:
                 dfact=5
              #st.write("You set alpha to:",alpha_value)  
              x_alpha = float(alpha_value)
              if x_alpha!=0:
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
                e = st.text_input('Please enter Y if you wish to download output files ')
                if e == 'Y':
                    sensor_data_update = pd.DataFrame()
                    sensor_data_update = sensor_data.copy()
                    sensor_data_update[x] = forecast_box_jen
                    csv = sensor_data_update.to_csv(index = False)
                    csv1 = outlier_df.to_csv(index = True)
                    b64 = base64.b64encode(csv.encode()).decode()
                    b641 = base64.b64encode(csv1.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="Box_Jenkins_Imputation.csv">Download</a>'
                    st.download_button(label='Download', data=csv, file_name='Box_Jenkins_Imputation.csv', mime='text/csv')
                    href1 = f'<a href="data:file/csv;base64,{b641}" download="Outlier.csv">Download outliers corrected in this imputation</a>'
                    st.download_button(label='Download outliers corrected in this imputation', data=csv1, file_name='Outlier.csv', mime='text/csv')
                else:
                    st.write("Thank you for using the application")
              else:
                 st.warning("Default value for alpha is 0.5")

            elif(option1=='Exponential Smoothening'):
              alpha_value = st.number_input("Enter an alpha value to be used by Exponential algorithm:")
              if not dfact:
                 dfact=5
              #st.write("You set alpha to:",alpha_value)  
              x_alpha = float(alpha_value)
              if x_alpha!=0:
                print("Exponential Smoothening Imputation in processing....")
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
                st.pyplot(fig)
                # sensor_data[x].plot(figsize=(20,10), title = "Original", color = ['#088f8f'])
                # df_forecast_exp_smooth.plot(figsize=(20, 10), title = "Exponential Smoothening Imputed", color = ['#088F8F'])
                ##Determining RMSE Value
                rmse_pandas = sqrt(mean_squared_error(sensor_data[x], df_forecast_exp_smooth))
                st.write('Exponential Smoothening Imputation RMSE :',rmse_pandas)
                ##Downloading Imputed Dataset
                f = st.text_input('Please enter Y if you wish to download output files ')
                if f == 'Y':
                    sensor_data_update = pd.DataFrame()
                    sensor_data_update = sensor_data.copy()
                    sensor_data_update[x] = forecast_exp_smooth
                    csv = sensor_data_update.to_csv(index = False)
                    csv1 = outlier_df.to_csv(index = True)
                    b64 = base64.b64encode(csv.encode()).decode()
                    b641 = base64.b64encode(csv1.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="Exp_Smoothening_Imputation.csv">Download</a>'
                    st.download_button(label='Download', data=csv, file_name='Exp_Smoothening_Imputation.csv', mime='text/csv')
                    href1 = f'<a href="data:file/csv;base64,{b641}" download="Outlier.csv">Download outliers corrected in this imputation</a>'
                    st.download_button(label='Download outliers corrected in this imputation', data=csv1, file_name='Outlier.csv', mime='text/csv')
                else:
                    st.write("Thank you for using the application")
              else:
                 st.warning("Default value for alpha is 0.5")
            else:
              st.write('Select valid option from dropdown')
        else:
          st.warning("This application provides 5 imputation algorithms options - MICE, KNN, Interpolation, Box Jenkins, and Exponential Smoothening")
    else:
        st.write('Thank You')


# else:
#    st.warning("Please upload a file")
# # view_boxplot(df.iloc[:,1:])
# # plt.show()