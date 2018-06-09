
# coding: utf-8

# In[1]:

get_ipython().system('jupyter nbconvert --to script !jupyter nbconvert --to script config_template.ipynb.ipynb')


# In[1]:

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt
import quandl
from stocker import Stocker
from sklearn import metrics
plt.rcParams['figure.figsize']=(20,10)
get_ipython().magic('matplotlib inline')


# ### Data Prep and Cleaning

# In[2]:

pg = Stocker(ticker ='PG') # Pulling data from the Quandl API, via Stocker package (open source). overkill, but easier than manually downloading, scraping, or calling the api.


# Dataset goes back to 1970 with daily price data for P&G stock.

# In[3]:

df_raw = pg.stock


# In[4]:

df_raw


# In[5]:

df_raw.head()


# In[6]:

df_pro = df_raw.reset_index().rename(columns={'Date':'ds', 'Close':'y'})
df_pro['y'] = np.log(df_pro['y'])


# In[7]:

df_pro.tail()


# In[ ]:




# In[23]:




# In[9]:

df_pro = df_pro.iloc[:,1:13]


# ### Running Prophet

# In[10]:

from fbprophet import Prophet


# In[11]:

model = Prophet(daily_seasonality = True, n_changepoints = 12) #specifying daily seasonality to see if there's any effect. This can be tweaked to weekly, or even monthly, if needed.
model.fit(df_pro)


# In[12]:

future = model.make_future_dataframe(periods=365) # forecasting for 1 year from now.
forecast = model.predict(future) #training the forecast model


# In[13]:

figure=model.plot(forecast)

# below is optional - If you want to plot y intercepts on the graph. 
#
#xcoords = model.changepoints
#for xc in xcoords:
#    plt.axvline(x=xc)


# Black line/dot = Actual data
# Blue line/dots = Forecast.
# Jumps in the graphs indicate stock splits

# In[14]:

model.plot_components(forecast)


# No clear conclusions here.
# 
# Daily vs ds plot
# 
# Daily data indicates hegihtened impacts over the weekends, but markets are closed over the weekends. Some after hours trading does happen, but nothing generally significant. Most importantly, activity during weekdays is flat 
# 
# Yearly vs ds plot
# 
# Seems like generally downward pressure on stock price in JFM , contrasted with upward pressure on price in OND for P&G.

# In[15]:

#Trying to understand what the errors might be; let's focus on the past 3 years instead.


# In[16]:

df_raw.head()


# In[17]:

df_raw2 = df_raw.rename(index=str, columns={"Close":"PG"})


# In[18]:

df_raw2


# In[19]:

df_raw3 = df_raw2[['Date', 'PG']]


# In[20]:

df_raw3 = df_raw3.rename(columns={'Date':'ds'})


# In[21]:

df_raw3 = df_raw3.set_index('ds')


# In[22]:

df_raw3.head()


# In[23]:

threeyr_data = forecast.set_index('ds').join(df_raw3) # building the forecast model.
threeyr_data = threeyr_data[['PG', 'yhat', 'yhat_upper','yhat_lower']].dropna().tail(1200)# pick the last 1200 entries, instead of th entirety of history of the stock. Also drop any blank data or NA.


# In[24]:

threeyr_data.head()


# In[25]:

threeyr_data.tail()


# In[26]:

threeyr_data['yhat']= np.exp(threeyr_data.yhat) #where exp is the exponential function. earlier we used logarithm to base e to get the predicted
threeyr_data['yhat_upper'] = np.exp(threeyr_data.yhat_upper) #  y values. Now we're bringing them up to predicted price value (dollars)
threeyr_data['yhat_lower'] = np.exp(threeyr_data.yhat_lower) 


# In[27]:

threeyr_data[['PG', 'yhat']].plot()


# Actual prices in blue
# Predicted prices in Orange
# 
# We could annalyse this further, but it's clear to see that the past 3 years experienced too much volatility.
# Looking at the initial graphs, it seems the last stock split was in 2004 (21st June 2004 to be precise.)
# After trying a few different starting points, I elected to limit going back to 10 years and start in 09 to cut out too much noise. This is still 2300 data points.
# So, let's train a new model going back to 2010 

# In[28]:

nostocksplits_data = forecast.set_index('ds').join(df_raw3) # building the forecast model.
nostocksplits_data = nostocksplits_data[['PG', 'yhat', 'yhat_upper','yhat_lower']].dropna().tail(2300)# pick the last 2300 entries


# In[29]:

nostocksplits_data.head()


# In[30]:

nostocksplits_data['yhat']= np.exp(nostocksplits_data.yhat) #where exp is the exponential function
nostocksplits_data['yhat_upper'] = np.exp(nostocksplits_data.yhat_upper) 
nostocksplits_data['yhat_lower'] = np.exp(nostocksplits_data.yhat_lower) 


# In[31]:

nostocksplits_data[['PG', 'yhat']].plot()


# Actual prices in blue. Prediction in orange.

# ## Basic Stats workup

# In[32]:

#to get an idea of how much data is missing. We should have 2018 Mar - 2009 Feb = 109 months of data. But below, we only have 77 rows. We're missing about 20 months of data; 20% gaps (on a daily basis)
nostocksplits_data.iloc[::7]


# In[33]:

#Average error over the past 13 years
tenyr_ae = (nostocksplits_data.yhat - nostocksplits_data.PG)
print (tenyr_ae.describe())


# In[34]:

#The numbers are not too bad; the forecasts on average deviate from the actual prices by ~20 cents, and a maximum (absolute) of 15 dollars at one point.
# Let's deeper dive into the numbers.


# In[35]:

#r2 comparisons
metrics.r2_score(nostocksplits_data.PG, nostocksplits_data.yhat)


# In[36]:

#R2 values less at 80 indicate an adequate model. We could improve the values with a tighter window,, but that's not the point here. An 80% score over 10 years is something I can work with.


# In[37]:

#explained variance regression score - Indexes the variance (square of standard deviation) of the difference between the forecast and actual data, with the variance of the actual data. Best possible score is 1.0; lower is worse.
metrics.explained_variance_score(nostocksplits_data.PG, nostocksplits_data.yhat)


# In[38]:

# Mean Absolute Error is the measurement of absolute error between two continuous variables. 
metrics.mean_absolute_error(nostocksplits_data.PG, nostocksplits_data.yhat)


# In[39]:

#medians are more robust to outliers. The loss is calculated by taking the median of all absolute differences between the target and the prediction.
metrics.median_absolute_error(nostocksplits_data.PG, nostocksplits_data.yhat)


# The median error between forecast and actual prices was 3.76 dollars. Not fantastic, but good enough to make long term plays on, given 3.76 dollars is roughly 3.76% of a stock price trending close to 100.

# ## Finishing up

# In[40]:

fig, ax1 = plt.subplots()
ax1.plot(nostocksplits_data.PG)
ax1.plot(nostocksplits_data.yhat)
ax1.plot(nostocksplits_data.yhat_upper, color='black',  linestyle=':', alpha=0.5)
ax1.plot(nostocksplits_data.yhat_lower, color='black',  linestyle=':', alpha=0.5)

ax1.set_title('Actual PG (Orange) vs PG Forecasted Upper & Lower Confidence (Black)')
ax1.set_ylabel('Price')
ax1.set_xlabel('Date')


# In[41]:

# Not exactly useful for intraday trading, but long term, even with lower confidence bounds, it seems 
# the trend is high and to the right, based on day-to-day price changes for the stock.


# In[42]:

df_raw_noss = df_pro.tail(2300) #no stock split dataset for date and price only


# In[43]:

df_raw_noss


# In[44]:

df_raw4 = df_raw3.tail(2300)


# In[45]:

# we have to build a new set of predictive models since we're only looking at the past 10 years now.

model1 = Prophet(daily_seasonality = True, n_changepoints = 12) #specifying daily seasonality to see if there's any effect. This can be tweaked to weekly, or even monthly, if needed.
model1.fit(df_raw_noss)
future1 = model1.make_future_dataframe(periods=365) # forecasting for 1 year from now.
forecast1 = model1.predict(future1) #training the forecast model


# In[113]:

#We've come this far, let's see what the model predicts anyway for the future, based on the past 10 years
full_df = forecast1.set_index('ds').join(df_raw4) #using the newly trained forecast1 model.
full_df['yhat']=np.exp(full_df['yhat'])

fig, ax1 = plt.subplots()
ax1.plot(full_df.PG)
ax1.plot(full_df.yhat, color='black', linestyle=':')
ax1.fill_between(full_df.index, np.exp(full_df['yhat_upper']), np.exp(full_df['yhat_lower']), alpha=0.5, color='darkgray')
ax1.set_title('Actual PG (Blue) vs PG Forecasted (Black) with Confidence Bands')
ax1.set_ylabel('Price')
ax1.set_xlabel('Date')

L=ax1.legend() #get the legend
L.get_texts()[0].set_text('PG Actal') #change the legend text for 1st plot
L.get_texts()[1].set_text('PG Forecasted') #change the legend text for 2nd plot


# In[47]:

full_df.tail()


# In[48]:

predicted_prices = full_df
predicted_prices['yhat_upper'] = np.exp(full_df.yhat_upper) 
predicted_prices['yhat_lower'] = np.exp(full_df.yhat_lower) 

predicted_prices.tail()


# ## Conclusion (Prophet)

# For the P&G stock price on 23rd March '19. Between 88 and 90 dollars is the target price prediction.

# # ARIMA model

# # ARIMA stands for Autoregressive Integrated Moving Average. Autoregressive models use previous values as predictors depending upon the form of the model and forecasts based on previous values,
# ### and are generally the benchmark for regression models. 

# In[49]:

df_pro.head()


# In[94]:

df_arima = df_raw3


# In[51]:

df_arima.head()


# In[88]:

df_arima.tail()


# In[52]:

from matplotlib import pyplot
df_arima.plot()
pyplot.show()


# In[53]:

## Taking a quick initial look at autocorrelation plots for the graph


# In[54]:

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(df_arima)
pyplot.show()


# In[55]:

### weak cyclical pattern observed. stronger correlation for first ~400 lags (above the no-correlation confidence intervals)
### The first 100-200 lags are more signifcant, so they may be a good starting point.


# In[56]:

# Building the ARIMA model
from statsmodels.tsa.arima_model import ARIMA 
#for the dataframe df_arima
# fit model
model = ARIMA(df_arima, order=(5,1,0)) #sets the lag value to 5 for autoregression, uses a diff order of 1 to make the time series statinoary, and uses a moving avg model of 0
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[57]:

from pandas import DataFrame
#plotting the residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# In[ ]:

# Strong residual errors where we had stock splits.


# In[58]:

df_arima.shape


# In[59]:

model_fit


# In[ ]:

# Similar to for prophet, let's only use the last 2300 values for the dataframe.


# In[64]:

from sklearn.metrics import mean_squared_error
X = df_arima.values                         # set X as the name of the dataframe in question
size = int(len(X) * 0.66)                   # setting up 66% train and 33% test sets 
train, test = X[0:size], X[size:len(X)]     #
history = [x for x in train]                 
#history = train
predictions = list()                        
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

#plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[ ]:

# Picking values for p and q in the ARIMA model


# In[ ]:

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff(), nlags=20)
lag_pacf = pacf(ts_log_diff(), nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff())),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff())),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[95]:

df_arima_09 = df_arima.tail(2300)


# In[96]:

#Now, we simply use all the data for the past ~10 years to train, and generate the coming 365 days of price data.

from sklearn.metrics import mean_squared_error
X = df_arima_09.values                         # set X as the name of the dataframe in question
train = X     
history = [x for x in train]                 
#history = train
predictions = history     
PG_365days = list()
for i in range(365):
    model = ARIMA(predictions, order=(4,0,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    PG_365days.append(yhat)
    #obs = test[t]
    #history.append(obs)
    print('predicted=%f, day=%f' % (yhat, i))
error = mean_squared_error(train, predictions)
print('Test MSE: %.3f' % error)

#plot
#pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[97]:

plt.plot(PG_365days)


# In[120]:

fig2, ax2 = plt.subplots()
ax2.plot(predictions, color = 'gray')
ax2.plot(train, linestyle=':')
ax2.set_title('Actual PG (Blue) vs PG Forecasted (Gray)')
ax2.set_ylabel('Price')
ax2.set_xlabel('Days since 2009')

L=ax1.legend() #get the legend
L.get_texts()[0].set_text('PG Actal') #change the legend text for 1st plot
L.get_texts()[1].set_text('PG Forecasted') #change the legend text for 2nd plot


# ## Thanks to

# In[ ]:

ericbrown - https://github.com/urgedata
Facebook Core Data Science team
towardsdatascience.com
Jason Brownlee (PhD) and the machinelearningmastery.com website
Analyticsvidhya.com


# ## Appendix
# ### Or, code that is nice to have.

# In[ ]:

# For the ARIMA model, after accounting for trends and seasonality (to make it a stationary model), the predictions became too conservative and essentially linear. The predicted price in a years time (21 March 2019)


# In[ ]:

len(test)


# #Setting the variables for the model
# - Number of AR (Auto-Regressive) terms (p): AR terms are just lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).
# - Number of MA (Moving Average) terms (q): MA terms are lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.
# - Number of Differences (d): These are the number of nonseasonal differences, i.e. in this case we took the first order difference. So either we can pass that variable and put d=0 or pass the original variable and put d=1. Both will generate same results.

# In[ ]:

df_arima.describe()


# In[ ]:

df_arima.median()


# In[ ]:

ts = df_arima


# In[ ]:

ts_log = np.log(ts)


# In[ ]:

plt.plot(ts_log, color = 'fuchsia')


# In[ ]:

moving_avg = ts_log.rolling(window = 12).mean()
plt.plot(ts_log, color = 'fuchsia')
plt.plot(moving_avg, color = 'yellow')


# In[ ]:

# Let's aim to make this time series stationary.
# The way to do that is by removing trends and seasonality.


# In[ ]:

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)


# In[ ]:

ts_log_moving_avg_diff.dropna(inplace=True)


# In[ ]:

# Testing for stationarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd =  timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    


# In[ ]:

# For the Dickey Fuller test, we'll have to separate the dataset into a single array.


# Dickey-Fuller Test: Statistical tests for stationarity. Here the null hypothesis is that the TS is non-stationary. The test results comprise of a Test Statistic and some Critical Values for different confidence levels. If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary.

# In[ ]:

ts_log_moving_avg_diff_dicky = ts_log_moving_avg_diff.iloc[:,0]


# In[ ]:

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(ts_dicky, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
     dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


# Test statistic is smaller than 1% Critical value, so we can say with 99% confidence that this time seiries is staionary.

# In[ ]:

ts_log.shift


# In[ ]:

ts_log_diff = ts_log - ts_log.shift # difference between adjacent values


# In[ ]:

ts_log_diff


# In[ ]:

plt.plot(ts_log_diff())


# In[ ]:

ts_log_diff()


# In[ ]:

# p – The lag value where the PACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case p=1.
# q – The lag value where the ACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case q=1.


# In[ ]:

#AR Model
model_AR = ARIMA(ts_log, order=(1, 1, 1))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff())
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff())**2))


# In[ ]:

from sklearn.metrics import mean_squared_error
X = df_arima.values                         # set X as the name of the dataframe in question
size = int(len(X) * 0.66)                   # setting up 66% train and 33% test sets 
train, test = X[0:size], X[size:len(X)]     #
history = [x for x in train]     
predictions = list()
i = 0
for i in range(1, 365): #the next 365 days after the end of this current dataset
    model_2019 = ARIMA(history, order=(1,1,1))
    model_fit_2019 = model_2019.fit(disp=0)
    output_1 = model_fit_2019.forecast()
    yhat = output_1[0]
    history.append(yhat)
    #obs = test[t]
    #history.append(obs) --> Necessary for the ARIMA model to keep going because it's a daily occurence.
    print('predicted=%f, day=%f' % (yhat, i))
error = mean_squared_error(days, predictions)
print('Test MSE: %.3f' % error)


# In[ ]:

output


# In[ ]:

pyplot.plot(test, color = 'yellow')
pyplot.plot(predictions, color = 'black')


# In[ ]:

pyplot.plot(predictions, color = 'green')


# In[ ]:

predictions[0]


# In[ ]:

# Following the Box-Jenkins methodology to fittting an ARIMA model.
# Model Identification
# Parameter estimation
# Model checking

