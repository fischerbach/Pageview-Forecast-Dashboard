import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from scipy import stats

st.title("Forecast Website Visits from choosen location")

st.header("Dataset overview")
dataset = pd.read_csv('dataset.csv')
dataset['day'] = pd.to_datetime(dataset['day'])
st.line_chart(dataset.groupby(by='day').sum())
dataset.set_index('day', inplace=True)

# st.header('Dataset overview')
# st.write(dataset[['city','unique users','page views']])

weather = pd.read_csv('weather.csv')
weather['day'] = pd.to_datetime(weather['day'])

st.header('Prediction')
city = st.selectbox('Select city', dataset['city'].unique())

df = dataset.query(f'city=="{city}"').copy()


df = df.merge(weather, left_on=['day', 'city'], right_on=['day','city']).set_index('day')
dummies = list(pd.get_dummies(df['text'], columns=['text'], prefix='', prefix_sep='').columns)
df = pd.get_dummies(df, columns=['text'], prefix='', prefix_sep='')

#Remove outliers
z_scores = stats.zscore(df[['page views']])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores >= 3).all(axis=1)
df['page views'][filtered_entries] = np.nan
# st.line_chart(df[['page views']])
start_date = st.date_input(
    'Select start date:', 
    df.index.max() - (df.index.max()-df.index.min())/2
)
train_date = st.date_input(
    'Select train set split date:', 
    df.index.max() - datetime.timedelta(days = 30)
)

#Cap and floor are set after outliers removal
df['floor'] = 0
df['cap'] = df['page views'].max()

st.write(df)
st.line_chart(df['page views'])

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation

def prophet_without_regressors(train, test):
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False, growth='linear')
    m.add_country_holidays(country_name='PL')

    m.fit(train)
    forecast = m.predict(test)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: 0 if x<0 else x)
    return forecast.set_index('ds')['yhat']

df['ds'] = df.index
df = df[start_date:] 
df['y'] = df['page views']
train = df[:train_date]
test = df[train_date:]
# test = df[:'2019']

# train['ds'] = train.index
# test['ds'] = test.index

# train['y'] = train['page views']
# test['y'] = test['page views']

# st.write(test['y'])

st.header("Prophet without regressors")
test['yhat'] = prophet_without_regressors(train, test)


st.line_chart(test[['y','yhat']])

st.header("Prophet with regressors")
available_regressors = ['maxtemp_c', 'mintemp_c', 'avgtemp_c', 'maxwind_kph', 'totalprecip_mm', 'avgvis_km', 'avghumidity', 'uv']
regressors = st.multiselect('Features',available_regressors+dummies, available_regressors)
def prophet_with_regressors(train, test):
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False, growth='linear')
    m.add_country_holidays(country_name='PL')
    for col in train.columns:
        if col in regressors and col not in ['y', 'cap', 'floor','Naive','Moving Average']:
            m.add_regressor(col)

    m.fit(train)

    forecast = m.predict(test)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: 0 if x<0 else x)
    
    return forecast.rename(columns={'yhat':'y'}).set_index('ds')['y']

test['yhat'] = prophet_with_regressors(train, test)

st.line_chart(test[['y','yhat']])

st.header('Evaluation')
from FES import FES

# def cross_validation(data, model=prophet_without_regressors):
#     date_range = (data['ds'].min()+datetime.timedelta(days=365), data['ds'].max())
#     datelist = pd.date_range(*date_range, freq='7D').tolist()
#     evaluation_statistics = []
#     for i in range(len(datelist)):
#         if (i == len(datelist) - 1):
#             continue
#         train_end = datelist[i]
#         test_start = datelist[i] + datetime.timedelta(days=1)
#         test_end = test_start + datetime.timedelta(days=5)
#         if(test_end > data['ds'].max()):
#             test_end = data['ds'].max()
#         else:
#             test_end = datelist[i] + datetime.timedelta(days=5)
#         train_end = datetime.datetime.strftime(train_end, '%Y-%m-%d')
#         test_start = datetime.datetime.strftime(test_start, '%Y-%m-%d')
#         test_end = datetime.datetime.strftime(test_end, '%Y-%m-%d')
        
#         print(f'{train_end} {test_start} {test_end}')
#         train = data[:train_end].copy()
#         test = data[test_start:test_end].copy()
#         train['ds'] = train.index
#         test['ds'] = test.index

#         evaluation_statistics.append([f":{train_end} | {test_start} - {test_end}",*FES.all(model(train, test), test['y'])])
#     evaluation_statistics = pd.DataFrame(evaluation_statistics, columns= ['period','ME', 'MSE', 'RMSE', 'MAE', 'MPE', 'MAPE', 'U1', 'U2'])
#     return evaluation_statistics

# # st.write(cross_validation(df))
# CV_evaluation = {}
# CV_evaluation['Prophet without regressors'] = cross_validation(df)
# CV_evaluation['Prophet with regressors'] = cross_validation(df, model = prophet_with_regressors)


# ma_vs_pr = pd.DataFrame(CV_evaluation['Prophet without regressors'].mean()).transpose()
# ma_vs_pr = ma_vs_pr.append(pd.DataFrame(CV_evaluation['Prophet with regressors'].mean()).transpose())

# ma_vs_pr['model'] = ['Prophet without regressors','Prophet with regressors']
# ma_vs_pr.set_index('model', inplace=True)
# st.write(ma_vs_pr)
# ma_vs_pr.to_html('cv.html')
evaluation_statistics = []
evaluation_statistics.append([f"Prophet without regressors",*FES.all(prophet_without_regressors(train, test), test['y'])])
evaluation_statistics.append([f"Prophet with regressors",*FES.all(prophet_with_regressors(train, test), test['y'])])
evaluation_statistics = pd.DataFrame(evaluation_statistics, columns= ['','ME', 'MSE', 'RMSE', 'MAE', 'MPE', 'MAPE', 'U1', 'U2'])
st.write(evaluation_statistics.set_index(''))