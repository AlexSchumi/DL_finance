
# coding: utf-8

# In[2]:

"""
This file is to download data from Google API
"""
import quandl
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[3]:

from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
# feed target data in dictionary
def get_dict(symbol):
    param = {
        'q': symbol, # Stock symbol (ex: "AAPL")
        'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
        'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
        'p': "15Y"  # Period (Ex: "1Y" = 1 year)
    }
    return(param)


# In[4]:

def get_price(symbol):
    param1 = {
        'q': symbol,  # Stock symbol (ex: "AAPL")
        'i': "86400",  # Interval size in seconds ("86400" = 1 day intervals)
        'x': "NASD",  # Stock exchange symbol on which stock is traded (ex: "NASD")
        'p': "15Y"  # Period (Ex: "1Y" = 1 year)
    }
    param2 = {
        'q': symbol,  # Stock symbol (ex: "AAPL")
        'i': "86400",  # Interval size in seconds ("86400" = 1 day intervals)
        'x': "NYSE",  # Stock exchange symbol on which stock is traded (ex: "NYSE")
        'p': "15Y"  # Period (Ex: "1Y" = 1 year)
    }
    df = get_price_data(param1)
    if df.empty:
        df = get_price_data(param2)
    return df


# In[5]:

"""
get_increase is to judge if the stock price has increased from one-day to another day

Input:
      d: close stock price of a ticker
      
Output: 
      increase: a list indicating whether the stock has a increase mode or not.     
"""

def get_increase(df):
    a = [x - df[i-1] if i > 0 else None for i, x in enumerate(df)][1:]
    increase = [1 if v > 0 else -1 for v in a]
    return(increase)  
"""
get_pattern is to find first occurance of specific pattern in stock to build database

Input:
      inc: increase list for stock
      start_point: entry point for another pattern search(for later use of search_all_patterns)
      
Output: 
      index pair: a index pair for consecutive increasing stock;
"""
def get_pattern(inc, start_point):
    for i in range(start_point, len(inc)):
        # look at the second-day increase stock
        if inc[i] == 1:
            add_val = sum(inc[i:i+5])
            if add_val == 5:
                return [i, i+4]
            # else:
                # continue


# In[109]:

"""
get_response is to process training response data, 0 for censored data, 1 for 10% increase stock price and -1 
for -10% decrease;

Input:
      st_price: stock_price of a ticker
      pattern_end: ending point for a pattern (for later use of search_all_patterns)
      
Output: 
      [days, response]
      days means the day when it get 10% increase or decrease
      response is 0, 1, -1 by definition;
"""
def get_response(st_price, pattern_end):
    rep = 0
    compare_end = pattern_end + 70 + 90 if (pattern_end + 70 + 90 < len(st_price)) else len(st_price)-1 # check the end of another window
    
    for i in range(pattern_end + 71, compare_end+1):
        if float(st_price[i] - st_price[pattern_end])/st_price[pattern_end] >= 0.1:
            rep = 1
            return [i-(pattern_end+70), rep] # return days of target and response;
        elif float(st_price[i] - st_price[pattern_end])/st_price[pattern_end] <= -0.1:
            rep = -1
            return [i-(pattern_end+70), rep]
        else:
            pass
    return [90, rep]    


# In[131]:

"""
search all patterns is to find all patterns within a stock

input: 
      df: stock_price for a symbol      
output:
      index for all entry points 
      response_day is corresponding get 10% up or down days
      response_index is whether 10% increase or decrease or censored data
"""
def search_all_patterns(st_price):
    start_point = 0
    index = []
    response_day = [] 
    response_index = []
    while(start_point <= (len(st_price)-75-90)):
        inc_ticker = get_increase(st_price)
        pattern = get_pattern(inc_ticker, start_point) # get pattern is to process inc_ticker list
        if pattern is not None:
            day = get_response(st_price, pattern[1])[0] # get response is to process raw data lists;
            res = get_response(st_price, pattern[1])[1]       
            index.append(pattern[0])
            response_day.append(day)
            response_index.append(res)
            start_point = pattern[1] + 10 # the new start of the start_point 
        else:
            break
    return index, response_day, response_index

