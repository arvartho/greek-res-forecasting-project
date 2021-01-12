import os
import json
import math
import random
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns
from fbprophet import Prophet
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

sns.set_theme(style="white")
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

class ModelEvaluation():
   def __init__(self, train_data=None, test_data=None):
      self.model_comparison_dict = {}
      self.config = json.loads(open('config.json', 'r').read())
      self.target_dates = self.config['target_dates']

      if train_data is not None and test_data is not None:
         self.train_data = train_data
         self.test_data = test_data

         print_df_specs(self.train_data, 'train set')
         print_df_specs(self.test_data, 'test set')

   # helper funtions
   def set_data(self, train_data, test_data):
      self.train_data = train_data
      self.test_data = test_data


   def ts_plot(self, x, y, title, ylim=None):
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=x, y=y, line_color='deepskyblue', opacity=.8))
      #fig.add_trace(go)
      fig.update_layout(title_text=title,
                  xaxis_rangeslider_visible=True)
      if ylim:
         fig.update_layout(yaxis=dict(range=ylim))
      fig.show()
      

   def ts_dualplot(self, x, y1, y2, title, l1='Line 1', l2='Line 2', color1=None, color2=None, opacity1=None, opacity2=None, ylim=None):
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=x, y=y1, name=l1, 
                              line_color=color1, opacity=opacity1))

      fig.add_trace(go.Scatter(x=x, y=y2, name=l2, 
                              line_color=color2, opacity=opacity2))

      #fig.add_trace(go)
      fig.update_layout(title_text=title,
                  xaxis_rangeslider_visible=True)   
      if ylim:
         fig.update_layout(yaxis=dict(range=ylim))
      fig.show()
      
   # plotting predicted vs actual energy values
   def plot_prediction(self, y_true, y_pred, model_name=None):
      """
      Plots the observed energy consumption against the predicted energy consumption
      """
      self.ts_dualplot(y_true.index,
                  y_true.values,
                  y_pred,
                  title='Observed vs predicted energy values using %s' % model_name,
                  l1='Observed',
                  l2='Predicted')
      
      #---------
      # Prediction vs observed energy correlation
      _, ax = plt.subplots(1,1, figsize=(5,5))
      ax.scatter(y_true.values, y_pred, alpha=.5) 
      _ = plt.xlabel("Observed energy")
      _ = plt.ylabel("Predicted energy")
      _ = plt.title("Observed vs Predicted energy correlation {}".format(model_name))
      # plotting 45 deg line to see how the prediction differs from the observed values
      x = np.linspace(*ax.get_xlim())
      _ = ax.plot(x, x, color='g')
      plt.show()

  
   def features_eval(self, feat_names, feat_values, model_name):
      # Plotting the coefficients to check the importance of each coefficient 
      # Plot the coefficients
      _ = plt.figure(figsize = (16, 7))
      _ = plt.plot(range(len(feat_names)), feat_values)
      _ = plt.xticks(range(len(feat_names)), feat_names, rotation = 85)
      _ = plt.margins(0.02)
      _ = plt.axhline(0, linewidth = 0.5, color = 'r')
      _ = plt.title('Feature importance of %s' % model_name)
      _ = plt.ylabel('lm_coeff')
      _ = plt.xlabel('Features')
      _ = plt.show()     

   # Metrics function
   def get_metrics(self, y_true, y_pred):
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      print('Root Mean Squared Error (RMSE): %.5f' % rmse)

      mae = mean_absolute_error(y_true, y_pred)
      print('Mean Absolute Error (MAE): %.5f' % mae)

      return {'MAE':np.round(mae, 5), 
            'RMSE':np.round(rmse, 5),
            'diff': np.abs(y_pred - y_true)}

   # wrapper function for model evaluation
   def model_eval(self, y_true, y_pred, model_name, plot=True):
      if plot:
         self.plot_prediction(y_true, y_pred, model_name = model_name)
      self.model_comparison_dict[model_name] = self.get_metrics(y_true, y_pred)


   def fbProphet_init(self, regressors, features):
      prophet = Prophet(
          growth='linear',
          daily_seasonality=False,
          weekly_seasonality=False,
          yearly_seasonality=False,
          changepoint_prior_scale=0.001,
          seasonality_mode='additive',
      )

      # Adding seasonalities
      if 'season_summer' in features:
         prophet.add_seasonality(
            name='summer', 
            period=6,
            fourier_order=2, 
            condition_name='season_summer')

      if 'season_winter' in features:
         prophet.add_seasonality(
            name='winter', 
            period=6, 
            fourier_order=2, 
            condition_name='season_winter')

      prophet.add_seasonality(
          name='daily',
          period=1,
          fourier_order=2,
      )

      prophet.add_seasonality(
          name='weekly',
          period=7,
          fourier_order=10,
      )

      prophet.add_seasonality(
          name='yearly',
          period=366,
          fourier_order=20,
      )

      # Adding external regressors
      for reg in regressors:
         prophet.add_regressor(reg, prior_scale=20, mode='additive', standardize='auto')

      return prophet


   def pfProphet_prediction(self, days_in_test, regressors, model_name, hybrid=False, hybrid_model=None, eval=True, metrics=True, plot=False, save=True):
      pred_dict = {'y_test':[], 'y_pred':[]}
      train_data = self.train_data.copy()
      train_data.loc[:, 'ds'] = train_data['ds'].dt.tz_localize(None)
      test_data = self.test_data.copy()
      test_data.loc[:, 'ds'] = test_data['ds'].dt.tz_localize(None)

      print("")
      print("Starting %s for week %s..." % (model_name, str(days_in_test[0])))
      print("")
      start = datetime.now()
      for test_day in days_in_test:
         # Prophet initialization
         prophet = self.fbProphet_init(regressors, train_data.columns)
         prophet.fit(train_data)

         _test_data = test_data.loc[test_data['ds'].dt.date==test_day].copy()
         merged_df = pd.concat([train_data, _test_data]).reset_index()
         y_pred_prophet = prophet.predict(merged_df)
         
         if hybrid:
            # Detrend data
            trend = y_pred_prophet.trend
            merged_df.loc[:, 'y'] = merged_df['y'] - trend
            
            # Data split for classic classifier
            x_train = merged_df.loc[merged_df['ds'].isin(train_data['ds']), :].set_index('ds').drop(columns=['y'])
            y_train = merged_df.loc[merged_df['ds'].isin(train_data['ds']), 'y']

            x_test = merged_df.loc[merged_df['ds'].isin(_test_data['ds']), :].set_index('ds').drop(columns=['y'])

            # Predict and add trend
            hybrid_model.fit(x_train, y_train)
            y_pred = hybrid_model.predict(x_test)
            y_pred = y_pred + (y_pred_prophet.loc[y_pred_prophet['ds'].isin(_test_data['ds']), 'trend'] ) 

         else:
            # Prediction correction
            # Add prediction lower and upper bounds at 0
            for col in y_pred_prophet.columns[1:]:
               y_pred_prophet.loc[y_pred_prophet[col] < train_data['y'].min(), col] = train_data['y'].min()
               y_pred_prophet.loc[y_pred_prophet[col] > train_data['y'].max(), col] = train_data['y'].max()
            
            y_pred = y_pred_prophet.set_index('ds').loc[_test_data['ds'], 'yhat']

         y_test = _test_data['y']

         if plot:
               self.plot_prediction(y_test, y_pred, model_name = '%s for %s' % (model_name, str(test_day)))
         if metrics:
            self.get_metrics(y_test, y_pred)

         pred_dict['y_pred'].append(y_pred)
         pred_dict['y_test'].append(y_test.values)         

         # Increament training data
         train_data = train_data.append(_test_data)  

      pred_dict['y_pred'] = np.hstack((pred_dict['y_pred']))
      pred_dict['y_test'] = np.hstack((pred_dict['y_test'])) 
      forecast_df = pd.DataFrame.from_dict(pred_dict)

      if eval: 
         self.rolling_eval(forecast_df, model_name)

      # Save prediction
      if save:
         self.save_forecast(forecast_df, model_name, days_in_test)

      print('Finished forecasting week %s in %s' % (str(days_in_test[0]), datetime.now()-start)) 
      return forecast_df


   def rolling_prediction(self, days_in_test, model, model_name, target_column, eval=True, metrics=True, plot=False, save=True):
      pred_dict = {'y_test':[], 'y_pred':[]}
      train_data = self.train_data.copy()
      test_data = self.test_data.copy()

      print("")
      print("Starting %s for week %s..." % (model_name, str(days_in_test[0])))
      print("")
      start = datetime.now()
      for test_day in days_in_test:

         x_train = train_data.copy()
         x_train.drop(columns=target_column, inplace=True)
         y_train = train_data.loc[:, target_column].copy()
         
         x_test = test_data.loc[test_data.index.date==test_day].copy()         
         x_test.drop(columns=target_column, inplace=True)
         y_test = test_data.loc[test_data.index.date==test_day, target_column].copy()         

         model.fit(x_train, y_train)
         y_pred = model.predict(x_test)     

         # Prediction correction: set lower bound at 0
         y_pred[y_pred<0] = 0  

         if plot:
            self.plot_prediction(y_test, y_pred, model_name = '%s for %s' % (model_name, str(test_day)))
         if metrics:
            self.get_metrics(y_test, y_pred)

         pred_dict['y_pred'].append(y_pred)
         pred_dict['y_test'].append(y_test.values)         

         # Increament training data
         train_data = train_data.append(test_data.loc[test_data.index.date==test_day])         

      pred_dict['y_pred'] = np.hstack((pred_dict['y_pred']))
      pred_dict['y_test'] = np.hstack((pred_dict['y_test'])) 
      forecast_df = pd.DataFrame.from_dict(pred_dict)

      if eval: 
         self.rolling_eval(forecast_df, model_name)

      # Save prediction
      if save:
         self.save_forecast(forecast_df, model_name, days_in_test)         

      print('Finished forecasting week %s in %s' % (str(days_in_test[0]), datetime.now()-start)) 
      return forecast_df

   
   def rolling_eval(self, forecast_df, model_name):
      model_name = 'Rolling %s' % model_name
      self.model_eval(forecast_df['y_test'], forecast_df['y_pred'], model_name)

   
   def execute_scenario(self, days_in_test, model, model_name, target_column, eval=True, plot=False):
      pass


   def plot_ts_decomp(self, decomp):   
      fig, ax = plt.subplots(4,1, figsize=(15,10))
      _ = ax[0].plot(decomp.observed)
      _ = ax[0].set_title('Observed')
      _ = ax[1].plot(decomp.trend)
      _ = ax[1].set_title('Trend')
      _ = ax[2].plot(decomp.seasonal)
      _ = ax[2].set_title('Seasonal')
      _ = ax[3].plot(decomp.resid)
      _ = ax[3].set_title('Residual')
      _ = fig.tight_layout()
      plt.show()


   def plot_corr_heatmap(self, df, features, interval=None, vmin=None, vmax=None):
      # Generate a mask for the upper triangle
      corr_mat = df[features].corr().round(2)
      mask = np.triu(np.ones_like(corr_mat, dtype=bool))

      # Set up the matplotlib figure
      _, _ = plt.subplots(figsize=(11, 9))

      # Generate a custom diverging colormap
      cmap = sns.diverging_palette(230, 20, as_cmap=True)

      # Draw the heatmap with the mask and correct aspect ratio
      sns.heatmap(corr_mat, mask=mask, cmap=cmap, annot= True, vmin=vmin, vmax=vmax,
                  square=True, linewidths=.5, cbar_kws={"shrink": .5})


   def save_forecast(self, df, model_name, days_in_test):
      energy_type = 'wind' if len([col for col in df.columns 
                                    if 'wind energy' in col.lower()])>1 else 'solar'
      week = str(days_in_test[0])

      forecast_dir = os.path.join('data', 'forecast_output', week)
      if not os.path.exists(forecast_dir):      
         os.makedirs(forecast_dir)
      df.to_csv(os.path.join(forecast_dir, '%s - %s.csv' % (energy_type, model_name)))
      

class Preprocessing():
   def __init__(self,):
      pass

   def final_feature_selection(self, train_data, target_column):
      from sklearn.feature_selection import RFECV
      from sklearn.model_selection import TimeSeriesSplit

      # Train-test split
      x_train = train_data.copy()
      x_train.drop(columns=target_column, inplace=True)
      y_train = train_data.loc[:, target_column].copy()

      # for time-series cross-validation set 5 folds
      tscv = TimeSeriesSplit(n_splits=5)
      # Train RFE feature selectior with cross validation
      selector = RFECV(xgb.XGBRegressor(), 
                     step=1, 
                     cv=tscv,
                     min_features_to_select = 10,
                     scoring='neg_mean_squared_error')

      selector.fit(x_train, y_train)
      features_selected = x_train.columns[selector.support_].tolist() + [target_column]
      return features_selected


hour_dict = {'morning': list(np.arange(7,13)),
             'afternoon': list(np.arange(13,16)), 
             'evening': list(np.arange(16,22)),
             'night': [22, 23, 0, 1, 2, 3, 4, 5, 6]}
def season_calc(month):
   if month in [6,7,8,9,10]:
     return "summer"
   else:
     return "winter"  

def time_of_day(h):
   if h in hour_dict['morning']:
      return 'morning'
   elif h in hour_dict['afternoon']:
      return 'afternoon'
   elif h in hour_dict['evening']:
      return 'evening'
   else:
      return 'night'

def add_temporal_features(df):
   df['hour'] = df.index.hour
   df['timeofday'] = df['hour'].apply(time_of_day)
   df['month'] = df.index.month
   df['year'] = df.index.year
   df['day'] = df.index.day
   df['dayofyear'] = df.index.dayofyear
   df['weekday_index'] = df.index.dayofweek
   df['season'] = df['month'].apply(lambda x: season_calc(x))
   return df

def floor_date(date):
   date = pd.to_datetime(date)
   return date - timedelta(hours=date.time().hour, 
                         minutes=date.time().minute, 
                         seconds=date.time().second, 
                         microseconds=date.time().microsecond)

def ceil_date(date):
   date = floor_date(date)   
   return date + timedelta(hours=23)

def print_df_specs(df, message=''):
   print('Shape of %s dataframe: %s,\nSize in memory (MB): %.2f' % (message, 
                                                                   df.shape, 
                                                                   df.memory_usage().sum()/1e6))





if __name__ == "__main__":
   import json
   from sklearn.linear_model import LinearRegression
   config = json.loads(open('config.json', 'r').read())
   target_dates = config['target_dates']
   rolling_eval = ModelEvaluation()

   for date in target_dates:
      # Load training data
      solar_energy_path = os.path.join('data', 'exported_data', str(date))
      train_data = pd.read_csv(os.path.join(solar_energy_path, 'solar_train.csv'), 
                              parse_dates=['time'], 
                              date_parser=lambda col: pd.to_datetime(col, utc=True),)
      train_data.loc[:, 'time'] = train_data.time.dt.tz_convert('CET')
      train_data.set_index('time', inplace=True)

      # Load testing data
      test_data = pd.read_csv(os.path.join(solar_energy_path, 'solar_test.csv'), 
                              parse_dates=['time'], 
                              date_parser=lambda col: pd.to_datetime(col, utc=True),)
      test_data.loc[:, 'time'] = test_data.time.dt.tz_convert('CET')
      test_data.set_index('time', inplace=True)

      days_in_test = sorted(set([date.date() for date in test_data.index]))
      rolling_eval.set_data(train_data, test_data)
      
      # Prediction
      clf_linreg = LinearRegression()
      rolling_pred_df = rolling_eval.rolling_prediction(days_in_test, 
                                                         clf_linreg, 
                                                         'Linear Regression for week %s' % str(date), 
                                                         'Solar energy (MW)', 
                                                         eval=True,
                                                         metrics=False,
                                                         plot=False,
                                                         save=False)
                                                      



   # ## Prophet
   # for date in target_dates:
   #    # Load training data
   #    wind_energy_path = os.path.join('data', 'exported_data', str(date))
   #    train_data = pd.read_csv(os.path.join(wind_energy_path, 'wind_train.csv'), 
   #                            parse_dates=['time'], 
   #                            date_parser=lambda col: pd.to_datetime(col, utc=True),)
   #    train_data.loc[:, 'time'] = train_data.time.dt.tz_convert('CET')
   #    train_data.set_index('time', inplace=True)

   #    # Load testing data
   #    test_data = pd.read_csv(os.path.join(wind_energy_path, 'wind_test.csv'), 
   #                            parse_dates=['time'], 
   #                            date_parser=lambda col: pd.to_datetime(col, utc=True),)
   #    test_data.loc[:, 'time'] = test_data.time.dt.tz_convert('CET')
   #    test_data.set_index('time', inplace=True)

   #    days_in_test = sorted(set([date.date() for date in test_data.index]))
   #    rolling_eval.set_data(train_data, test_data)
   #    rolling_pred_df = rolling_eval.rolling_prediction(days_in_test, 
   #                                                       LinearRegression(), 
   #                                                       'Linear regression',  
   #                                                       'Wind energy (MW)', 
   #                                                       eval=True,
   #                                                       plot=False)


   # ## Prophet
   # for date in target_dates:
   #    # Load training data
   #    wind_energy_path = os.path.join('data', 'exported_data', str(date))
   #    train_data = pd.read_csv(os.path.join(wind_energy_path, 'wind_train.csv'), 
   #                            parse_dates=['time'], 
   #                            date_parser=lambda col: pd.to_datetime(col, utc=True),)
   #    train_data.loc[:, 'time'] = train_data.time.dt.tz_convert('CET')
   #    train_data.rename(columns = {'Wind energy (MW)':'y', 'time':'ds'}, inplace=True)

   #    # Load testing data
   #    test_data = pd.read_csv(os.path.join(wind_energy_path, 'wind_test.csv'), 
   #                            parse_dates=['time'], 
   #                            date_parser=lambda col: pd.to_datetime(col, utc=True),)
   #    test_data.loc[:, 'time'] = test_data.time.dt.tz_convert('CET')
   #    test_data.rename(columns = {'Wind energy (MW)':'y', 'time':'ds'}, inplace=True)

   # days_in_test = sorted(set(test_data['ds'].dt.date.values))
   # rolling_eval.set_data(train_data, test_data)

   # rolling_pred_df = rolling_eval.pfProphet_prediction(days_in_test, 
   #                                                       ['windSpeed', 'gust', 'lagged_energy'],
   #                                                       'FB Prophet for week %s' % str(date), 
   #                                                       hybrid=True, 
   #                                                       hybrid_model=None,
   #                                                       eval=True, 
   #                                                       metrics=True, 
   #                                                       plot=False)

   ## Hybrid
   for date in target_dates:
      # Load training data
      wind_energy_path = os.path.join('data', 'exported_data', str(date))
      train_data = pd.read_csv(os.path.join(wind_energy_path, 'wind_train.csv'), 
                              parse_dates=['time'], 
                              date_parser=lambda col: pd.to_datetime(col, utc=True),)
      train_data.loc[:, 'time'] = train_data.time.dt.tz_convert('CET')
      train_data.rename(columns = {'Wind energy (MW)':'y', 'time':'ds'}, inplace=True)

      # Load testing data
      test_data = pd.read_csv(os.path.join(wind_energy_path, 'wind_test.csv'), 
                              parse_dates=['time'], 
                              date_parser=lambda col: pd.to_datetime(col, utc=True),)
      test_data.loc[:, 'time'] = test_data.time.dt.tz_convert('CET')
      test_data.rename(columns = {'Wind energy (MW)':'y', 'time':'ds'}, inplace=True)

      days_in_test = sorted(set(test_data['ds'].dt.date.values))
      rolling_eval.set_data(train_data, test_data)
      
      # Tuning parameters
      param_grid = {
         'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
         'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
         'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
         'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
         'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
         'gamma': [0, 0.25, 0.5, 1.0],
         'n_estimators': [10, 31, 52, 73, 94, 115, 136, 157, 178, 200]}

      # for time-series cross-validation set 5 folds
      tscv = TimeSeriesSplit(n_splits=5)

      clf_xgb = RandomizedSearchCV(xgb.XGBRegressor(),
                                 param_distributions=param_grid,
                                 scoring='neg_mean_squared_error',
                                 n_iter=20,
                                 n_jobs=-1, 
                                 cv=tscv,
                                 verbose=0,
                                 random_state=42)

      rolling_pred_df = rolling_eval.pfProphet_prediction(days_in_test, 
                                                            ['windSpeed', 'gust', 'lagged_energy'],
                                                            'Hybrid FB Prophet and Extreme Gradient Boost for week %s' % str(date), 
                                                            hybrid=True,
                                                            hybrid_model=clf_xgb,
                                                            eval=True, 
                                                            metrics=True, 
                                                            plot=False)

      

