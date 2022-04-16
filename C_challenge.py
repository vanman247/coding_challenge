import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

def gq_1_example():
    base_url = 'https://github.com/byuidatascience/data4missing/'
    flights_path = 'raw/master/data-raw/flights_missing/flights_missing.json'
    url_flights = base_url + flights_path

    df = (pd.read_json(url_flights)
        .replace({"month":{"n/a":np.nan}})
        .dropna(subset=['month']))
    print(df.info())
    return df

#gq_1_example()

def gq_1_answer():
    base_url = 'https://github.com/byuidatascience/data4missing/'
    flights_path = 'raw/master/data-raw/flights_missing/flights_missing.json'
    url_flights = base_url + flights_path

    df = pd.read_json(url_flights)

    df = df[df["month"] != "n/a"]
    print(df.info())
    return df

# gq_1_answer()

def GQ_2():
    base_url = 'https://github.com/byuidatascience/data4missing/'
    flights_path = 'raw/master/data-raw/flights_missing/flights_missing.json'
    url_flights = base_url + flights_path

    df = (pd.read_json(url_flights)
        .replace({"month":{"n/a":np.nan}})
        .dropna(subset=['month']))
    ## 2 columns are minutes_delayed_nas & minutes_delayed_carrier, because they have lower numbers then the rest of the "Minutes" data.

    df["minutes_delayed_nas"].fillna(float(df['minutes_delayed_nas'].mean()), inplace=True)
    df["minutes_delayed_carrier"].fillna(float(df['minutes_delayed_carrier'].mean()), inplace=True)

    df = df[["minutes_delayed_nas", "minutes_delayed_carrier"]]
    print(df.info())
    print(df.head())

    # Now Find the Standard Deviation of these 2 columns
    print("\n \n Standard Deviation of Minutes delayed NAS & Carrier: \n", round(df.std(), 2))

    return df

GQ_2()

def GQ_3():
    base_url = 'https://github.com/byuidatascience/data4missing/'
    flights_path = 'raw/master/data-raw/flights_missing/flights_missing.json'
    url_flights = base_url + flights_path

    df = pd.read_json(url_flights)

    month_dict = {"January":1, "Febuary":2, "March":3, "April":4, 
    "May":5, "June":6, "July":7, "August":8, "September":9, 
    "October":10, "November":11, "December":12}

    df = df[df["month"] != "n/a"]

    df = df["month"].replace(month_dict, method='bfill').astype("int64")

    print("Median Month in the Data Set is the ", round(df.mean(), 0), "th")
    return df

GQ_3()

def GQ_4():
    base_url = 'https://github.com/byuidatascience/data4missing/'
    flights_path = 'raw/master/data-raw/flights_missing/flights_missing.json'
    url_flights = base_url + flights_path

    df = pd.read_json(url_flights)

    df["num_of_delays_carrier"] =  df["num_of_delays_carrier"].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
    df["num_of_delays_carrier"].astype("float64")
    print(df.info())
    return df

GQ_4()