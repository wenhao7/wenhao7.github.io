---
layout: post
title:  "COVID-19 Around The World"
date:   2024-01-01 22:31:08 +0800
category: [data_wrangling, data_analysis, visualization]
tag: [numpy, pandas, seaborn, matplotlib, sql, statistics]
summary: "In this notebook we will be analysing and discussing Covid-19 related data from all around the world, looking at how the pandemic hits different places differently and how to understand some statistics commonly quoted on mainstream/social media."
---

***

## Contents
1. [Overview](#1)
2. [Data](#2)
3. [Global Stats](#3)
4. [Continental Stats](#4)
5. [Country Stats](#5)<br>
    5.1 [Cases/Deaths](#5.1)<br>
    5.2 [Fatality Rates](#5.2)<br>
    5.3 [Vaccinations](#5.3)<br>
    5.4 [Testings](#5.4)<br>
    5.5 [Excess Mortalities](#5.5)<br>
    5.6 [Hospitalisations](#5.6)<br>
    5.7 [Other Data](#5.7)
6. [Conclusion](#6)    
    
***

<a id = '1'></a>
## 1. Overview
In this notebook we will be analysing and discussing Covid-19 related data from all around the world, looking at how the pandemic hits different places differently and how to understand some statistics commonly quoted on mainstream/social media.

***

<a id = '2'></a>
## 2. Data
The data we will be using in this notebook can be found [here](https://github.com/owid/covid-19-data/tree/master/public/data). An [SQL database](https://github.com/wenhao7/Data-Science/blob/main/Covid%20Around%20The%20World/create_database.py) containing the data was created and data is queried into pandas dataframes as needed in each section.


```python
import psycopg2 as ps
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import Error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings('ignore')
```


```python
# Function to execute queries and save data in a dataframe
def query(sql1):
    con = ps.connect(user = 'postgres',
                 password = '0000',
                 database = 'our_world'
                 )

    cur = con.cursor()
    
    try:
        cur.execute(sql1)
        #print(cur.fetchall())
        
        return cur.fetchall()
        
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)


# Function to extract column names from executed queries    
def column_names(sql1):
    con = ps.connect(user = 'postgres',
                 password = '0000',
                 database = 'our_world'
                 )

    curs = con.cursor()
    
    try:    
        curs.execute(sql1)
              
        colnames = [desc[0] for desc in curs.description]
        return colnames
    
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
```

***

<a id = '3'></a>
## 3. Global Stats
In this section we will look at global data during the pandemic.


```python
#       Number of confirmed cases / deaths out of global population (over time)
q = """SELECT date, SUM(total_cases), SUM(new_cases), 
        SUM(new_cases_smoothed), SUM(total_deaths),
        SUM(new_deaths), SUM(new_deaths_smoothed) 
        FROM cases
        GROUP BY date"""
colnames = ['date', 'total_cases', 'new_cases', 'new_cases_smoothed'
                , 'total_deaths','new_deaths','new_deaths_smoothed']
df = pd.DataFrame(data = query(q), columns = colnames)
                              
df.date = pd.to_datetime(df.date)
```


```python
# New Cases Over Time                       
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (12,24))
ax1.set_title('Global Daily New Confirmed Cases', fontsize = 22)
ax1.bar(x = df.date, height = df.new_cases, width = 1, color = 'lightseagreen')
ax1.plot(df.date, df.new_cases_smoothed, color = 'orange')

handles, labels = ax1.get_legend_handles_labels()
patch = mpatches.Patch(color = 'lightseagreen', label = 'Daily New Cases')
patch1 = mpatches.Patch(color = 'orange', label = '7 Day Average')
handles.append(patch)
handles.append(patch1)
ax1.legend(handles = handles, loc = 'upper left')

ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('New Cases', fontsize = 18)


# Total Cases Over Time (Linear y scale)
ax2.set_title('Global Confirmed Cases Over Time (Linear)', fontsize = 22)
ax2.bar(x = df.date, height = df.total_cases, width = 1, color = 'teal')

handles, labels = ax2.get_legend_handles_labels()
patch = mpatches.Patch(color = 'teal', label = 'Total Cases')
handles.append(patch)
ax2.legend(handles = handles, loc = 'upper left')

ax2.set_xlabel('Date', fontsize = 18)
ax2.set_ylabel('Total Cases', fontsize = 18)  


# Total Cases Over Time (Logarithmic y scale)
ax3.set_title('Global Confirmed Cases Over Time (Log)', fontsize = 22)
ax3.bar(x = df.date, height = df.total_cases, width = 1, color = 'teal')

handles, labels = ax3.get_legend_handles_labels()
patch = mpatches.Patch(color = 'teal', label = 'Total Cases')
handles.append(patch)
ax3.legend(handles = handles, loc = 'upper left')

plt.yscale('log')
ax3.set_xlabel('Date', fontsize = 18)
ax3.set_ylabel('Total Cases', fontsize = 18)

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_5_0.png)
    


In the first plot above we see there has been three major waves of infections across the world since the start of the pandemic. These figures are affected more heavily by countries/regions with higher populations so it may not be representative of what most countries have actually experienced during the pandemic.

When looking at cumulative confirmed cases there are typically two ways to plot the time-series data, in a linear or a logarithmic manner. These two options differs in several ways.

The biggest advantage for linear graphs is that most people will know how to read the graph. On the other hand, a linear scale will dramatize the increase in the number of cases emphasizing the exponential "explosion" in the spread of infection. This makes identifying the change in rate of spread harder in a linear graph.

When looking at a logarithmic graph the main problem will be the worry of whether ordinary citizens have the ability to properly interpret them. By looking at the slope of the curve on a log-scale, one is able to easily assess the rate at which the pandemic is spreading. An exponential growth rate will appear as a straightline on a log graph, in the graph above we can see areas where the slope decreases suggesting a reducing growth rate. 

In terms of effect on individual opinion on public health measures when looking at these plots, a linear graph may present a more pessimistic perspective on the pandemic due to the more dramatic growth in number of cases, while a logarithmic graph may present a more optimistic perspective where the growth is under control.


```python
# New Deaths Over Time                       
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (12,24))
ax1.set_title('Global Daily New Deaths', fontsize = 22)
ax1.bar(x = df.date, height = df.new_deaths, width = 1, color = 'darkslategrey', alpha = 0.8)
ax1.plot(df.date, df.new_deaths_smoothed, color = 'red')

handles, labels = ax1.get_legend_handles_labels()
patch = mpatches.Patch(color = 'darkslategrey', label = 'Daily New Deaths')
patch1 = mpatches.Patch(color = 'red', label = '7 Day Average')
handles.append(patch)
handles.append(patch1)
ax1.legend(handles = handles, loc = 'upper left')

ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('New Deaths', fontsize = 18)


# Total Cases Over Time (Linear y scale)
ax2.set_title('Global Confirmed Deaths Over Time (Linear)', fontsize = 22)
ax2.bar(x = df.date, height = df.total_deaths, width = 1, color = 'darkslategrey')

handles, labels = ax2.get_legend_handles_labels()
patch = mpatches.Patch(color = 'darkslategrey', label = 'Total Deaths')
handles.append(patch)
ax2.legend(handles = handles, loc = 'upper left')

ax2.set_xlabel('Date', fontsize = 18)
ax2.set_ylabel('Total Deaths', fontsize = 18)  


# Total Cases Over Time (Logarithmic y scale)
ax3.set_title('Global Confirmed Deaths Over Time (Log)', fontsize = 22)
ax3.bar(x = df.date, height = df.total_deaths, width = 1, color = 'darkslategrey')

handles, labels = ax3.get_legend_handles_labels()
patch = mpatches.Patch(color = 'darkslategrey', label = 'Total Deaths')
handles.append(patch)
ax3.legend(handles = handles, loc = 'upper left')

plt.yscale('log')
ax3.set_xlabel('Date', fontsize = 18)
ax3.set_ylabel('Total Deaths', fontsize = 18)

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_7_0.png)
    


When looking at confirmed deaths over time the graphs look very similar to confirm cases over time except for the first graph showing global daily new deaths, where an extra peak is seen at the very start of the pandemic.


```python
# New Cases and Deaths Over Time
fig, ax1 = plt.subplots(figsize = (12,8))
ax1.set_title('Global Daily New Confirmed Cases and Deaths', fontsize = 22)
ax1.bar(x = df.date, height = df.new_cases, width = 1, color = 'lightseagreen', alpha = 0.2)
ax1.plot(df.date, df.new_cases_smoothed, color = 'orange')

ax2 = ax1.twinx()
ax2.bar(x = df.date, height = df.new_deaths, width = 1, color = 'darkslategrey', alpha = 0.2)
ax2.plot(df.date, df.new_deaths_smoothed, color = 'red')

handles, labels = ax1.get_legend_handles_labels()
patch = mpatches.Patch(color = 'lightseagreen', label = 'Daily New Cases')
patch1 = mpatches.Patch(color = 'darkslategrey', label = 'Daily New Deaths')
patch2 = mpatches.Patch(color = 'orange', label = '7 Day Average Cases')
patch3 = mpatches.Patch(color = 'red', label = '7 Day Average Deaths')
handles.append(patch)
handles.append(patch1)
handles.append(patch2)
handles.append(patch3)
ax1.legend(handles = handles, loc = 'upper left')

ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('New Cases', fontsize = 18)
ax2.set_ylabel('New Deaths', fontsize = 18)
```




    Text(0, 0.5, 'New Deaths')




    
![png](/images/covid_around_world/output_9_1.png)
    


When plotting the global daily cases and deaths together we observe an extra peak in daily deaths at the start of the pandemic. This can be attributed to the virus being unfamiliar to the world at that point in time leading to higher fatality rates. As time goes by, knowledge on handling the infections and the logistics/equipment needed to support treatment are set up leading to better survival rates. 

The latest peak for daily deaths being lower than the ones before may possibly be due to vaccination rates around the world increasing, significantly lowering the death rate of vaccinated people who gets infected.

***

<a id = '4'></a>
## 4. Continental Stats
In this section the global data shall be separated into their respective continents.


```python
#       Number of confirmed cases / active cases / deaths out of global population over time
q = """SELECT cases.date, others.continent, SUM(cases.total_cases), SUM(cases.new_cases), 
        SUM(cases.new_cases_smoothed), SUM(cases.total_deaths),
        SUM(cases.new_deaths), SUM(cases.new_deaths_smoothed) 
        FROM cases
        JOIN others ON cases.iso_code = others.iso_code
        GROUP BY cases.date, others.continent"""
df = pd.DataFrame(data = query(q),
columns = ['date', 'continent','total_cases','new_cases','new_cases_smoothed',
           'total_deaths','new_deaths','new_deaths_smoothed']
)
                  
df.date = pd.to_datetime(df.date)
df = df.loc[df.continent.notnull()]     
```


```python
# New Cases Over Time                       
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (12,8))
ax1.set_title('Continental Daily New Cases', fontsize = 22)
sns.lineplot(x = 'date', y = 'new_cases_smoothed', data = df, hue = 'continent', ax = ax1)

ax1.legend()

ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('New Cases', fontsize = 18)

# Total Cases Over Time
ax2.set_title('Continental Confirmed Cases Over Time', fontsize = 22)
sns.lineplot(x = 'date', y = 'total_cases', data = df, hue = 'continent', ax = ax2)

ax2.legend()

plt.yscale('log')
ax2.set_xlabel('Date', fontsize = 18)
ax2.set_ylabel('Total Cases', fontsize = 18)
plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_12_0.png)
    


From the graph above we see that the trends for each continent differs from the trends seen in the global data. Continents such as North America, Europe, and Asia has seen distinct waves in daily new cases while continents like Oceania and South America has not seen distinct waves like the other continents.


```python
# New Death Over Time                       
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12,8))
ax1.set_title('Continental Daily New Deaths', fontsize = 22)
sns.lineplot(x = 'date', y = 'new_deaths_smoothed', hue = 'continent', data = df, ax = ax1)

ax1.legend()

ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('New Deaths', fontsize = 18)

# Total Deaths Over Time
ax2.set_title('Continental Confirmed Deaths Over Time', fontsize = 22)
sns.lineplot(x = 'date', y = 'total_deaths', data = df, hue = 'continent', ax = ax2)

ax2.legend()

plt.yscale('log')
ax2.set_xlabel('Date', fontsize = 18)
ax2.set_ylabel('Total Deaths', fontsize = 18)

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_14_0.png)
    


In terms of daily deaths over time we observe the same trend as the global graph, where continents where the infection spread early in the pandemic has high fatality rates in that period of time.

***

<a id = '5'></a>
## 5. Countries Stats
In this section we will be looking at data from a group of countries in the dataset that have relatively complete data and are spread across all continents.

<a id = '5.1'></a>
### 5.1 Cases / Deaths
First, we shall look at number of cases / deaths.


```python
#       Number of daily (weekly) new cases / new deaths over time (per xxxx population)
q = """SELECT date, iso_code, location, total_cases, new_cases, 
        new_cases_smoothed, total_deaths,
        new_deaths, new_deaths_smoothed,
        total_cases_per_million, new_cases_per_million,
        new_cases_smoothed_per_million,
        total_deaths_per_million, new_deaths_per_million,
        new_deaths_smoothed_per_million
        FROM cases
        WHERE iso_code IN ('USA','GBR','FRA','ITA','CAN','BRA','BEL','ISR',
                           'NLD','TWN','KOR','SGP','PHL','IND','AUS','PNG')
"""
colnames = column_names(q)
df = pd.DataFrame(data = query(q), columns = colnames)

df.date = pd.to_datetime(df.date)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>iso_code</th>
      <th>location</th>
      <th>total_cases</th>
      <th>new_cases</th>
      <th>new_cases_smoothed</th>
      <th>total_deaths</th>
      <th>new_deaths</th>
      <th>new_deaths_smoothed</th>
      <th>total_cases_per_million</th>
      <th>new_cases_per_million</th>
      <th>new_cases_smoothed_per_million</th>
      <th>total_deaths_per_million</th>
      <th>new_deaths_per_million</th>
      <th>new_deaths_smoothed_per_million</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-26</td>
      <td>AUS</td>
      <td>Australia</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.155</td>
      <td>0.155</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-27</td>
      <td>AUS</td>
      <td>Australia</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.194</td>
      <td>0.039</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-28</td>
      <td>AUS</td>
      <td>Australia</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.194</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-29</td>
      <td>AUS</td>
      <td>Australia</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.233</td>
      <td>0.039</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-30</td>
      <td>AUS</td>
      <td>Australia</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.349</td>
      <td>0.116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# New Cases Over Time                       
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (12,24))
ax1.set_title("Countries' Daily New Cases", fontsize = 22)
sns.lineplot(x = 'date', y = 'new_cases_smoothed', data = df, hue = 'location', ax = ax1)

ax1.legend()

ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('New Cases', fontsize = 18)

# New Cases per Million Over Time          
ax2.set_title("Countries' Daily New Confirmed Cases Per Million", fontsize = 22)
sns.lineplot(x = 'date', y = 'new_cases_smoothed_per_million', data = df, hue = 'location', ax = ax2)

ax2.legend()

ax2.set_xlabel('Date', fontsize = 18)
ax2.set_ylabel('New Cases', fontsize = 18)

# Total Cases Over Time
ax3.set_title("Countries' Confirmed Cases Over Time", fontsize = 22)
sns.lineplot(x = 'date', y = 'total_cases', data = df, hue = 'location', ax = ax3)

ax3.legend()

plt.yscale('log')
ax3.set_xlabel('Date', fontsize = 18)
ax3.set_ylabel('Total Cases', fontsize = 18)

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_17_0.png)
    


When looking at daily new cases data for the various countries, confirmed cases per million population is a good way to present the data in a way that normalizes the difference in the populations between the countries.

All three graphs shows the countries having distinct waves of infections. In the first two linear graphs the individual peaks in number of daily new cases are self-explanatory while in the third graph the waves can be identified at regions where the slope of the curves experience sharp increases.


```python
# New Death Over Time                       
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (12,24))
ax1.set_title("Countries' Daily New Deaths", fontsize = 22)
sns.lineplot(x = 'date', y = 'new_deaths_smoothed', hue = 'location', data = df, ax = ax1)

ax1.legend()

ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('New Deaths', fontsize = 18)

# New Cases per Million Over Time     
ax2.set_title("Countries' Daily New Deaths Per Million", fontsize = 22)
sns.lineplot(x = 'date', y = 'new_deaths_smoothed_per_million', data = df, hue = 'location', ax = ax2)

ax2.legend()

ax2.set_xlabel('Date', fontsize = 18)
ax2.set_ylabel('New Cases', fontsize = 18)

# Total Deaths Over Time
ax3.set_title("Countries' Confirmed Deaths Over Time", fontsize = 22)
sns.lineplot(x = 'date', y = 'total_deaths', data = df, hue = 'location', ax = ax3)

ax3.legend()

plt.yscale('log')
ax3.set_xlabel('Date', fontsize = 18)
ax3.set_ylabel('Total Deaths', fontsize = 18)

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_19_0.png)
    


When looking at daily deaths, we see the same pattern observed before. For most countries the first wave of infection (relatively small absolute number of infections) leads to the highest fatality rates of the pandemic. This is most visible in Belgium where the first wave of infections saw 10 times less confirmed cases per million, but twice as many deaths per million than the second wave.


<a id = '5.2'></a>
### 5.2 Fatality Rates
In this section we will look at common metrics used to describe the fatality rates of the virus.


```python
# Fatality Rates
q = """SELECT date, cases.iso_code, cases.location, total_cases, new_cases, 
        new_cases_smoothed, total_deaths,
        new_deaths, new_deaths_smoothed,
        total_cases_per_million, new_cases_per_million,
        new_cases_smoothed_per_million,
        total_deaths_per_million, new_deaths_per_million,
        new_deaths_smoothed_per_million,
        others.population
        FROM cases
        JOIN others
        ON cases.iso_code = others.iso_code
        WHERE cases.iso_code IN ('USA','GBR','FRA','ITA','CAN','BRA','BEL','ISR',
                           'NLD','TWN','KOR','SGP','PHL','IND','AUS','PNG')
"""
colnames = column_names(q)
df = pd.DataFrame(data = query(q), columns = column_names(q))

df.date = pd.to_datetime(df.date)

df['cfr'] = df['total_deaths']/df['total_cases'] * 100 # Case Fatality Rate
df['cmr'] = df['total_deaths']/df['population'] * 100 # Crude Mortality Rate
```

**Case fatality rate (CFR)** is defined as total number of deaths divided by the total number of confirmed cases. The problem with this metric is that the total number of confirmed cases typically misses out a significant portion of actual cases especially with Covid-19 where asymptomic infections are common. Depending on how widespread testing is (how many infections are tested and confirmed) the case fatality rate may present a number higher than what the actual fatality rate is.

**Crude mortality rate (CMR)** is defined as total number of deaths divided by total population. This metrics' main problem is that it assumes everyone in the population has been infected which is not the case. This can result in a number significantly lower than the actual fatality rate downplaying the severity of the virus.


```python
#       Case fatality rate (overtime)(per xxxx population) <--- Typical metric discussed for mortality, misleading as confirmed cases not total cases
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12,16))
ax1.set_title("Case Fatality Rate Over Time", fontsize = 22)
sns.lineplot(x = 'date', y = 'cfr', data = df, hue = 'location', ax = ax1)

ax1.legend()

ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('Case Fatality Rate (%)', fontsize = 18)

#       Crude mortality rate (per xxxx population) histogram <--- Another metrics used to downplay covid, misleading as total pop not total cases
ax2.set_title("Crude Mortality Rate Over Time", fontsize = 22)
sns.lineplot(x = 'date', y = 'cmr', data = df, hue = 'location', ax = ax2)

ax2.legend()

ax2.set_xlabel('Date', fontsize = 18)
ax2.set_ylabel('Crude Mortality Rate (%)', fontsize = 18)

plt.tight_layout()
plt.show()   
```


    
![png](/images/covid_around_world/output_23_0.png)
    


When looking at how CFR changes throughout the pandemic, some countries experienced a high CFR at the earlier stages of the pandemic, going as high as 15% for a few months in some cases. Currently in October 2021, the CFR of most countries falls somewhere below 2%.

When looking at CMR, we see that it grows in a similar manner to the total deaths over time. As a metric, CMR will only continue to grow over time assuming the same population size and if we can be sure that almost everyone in a population has been infected then the CMR will be able to reflect the true fatality rate of the virus.

Hence, when discussing the severity infection one has to beware of touted survival rates of 99.8% floating around which typically comes from the CMR of a country. This figure grossly underestimates the fatality rates of the virus and may give people a false sense of security. A better figure for a layperson to use would be the CFR which will provide a better picture of how dangerous the virus is, at least until everyone in a population as been infected at least once.

<a id = '5.3'></a>
### 5.3 Vaccinations
In this section we shall look at data relating to vaccinations across the countries.


```python
## Vaccinations
q = """SELECT vaccinations.*, others.population FROM vaccinations
JOIN others
ON vaccinations.iso_code = others.iso_code
WHERE vaccinations.iso_code IN ('USA','GBR','FRA','ITA','CAN','BRA','BEL','ISR',
                       'NLD','TWN','KOR','SGP','PHL','IND','AUS','PNG')"""      
colnames = column_names(q)

df = pd.DataFrame(data = query(q),
                  columns = colnames)

df['vaccinated'] = df['people_vaccinated']/df['population'] * 100
df['fully_vaccinated'] = df['people_fully_vaccinated']/df['population'] * 100
df['boostered'] = df['total_boosters']/df['population'] * 100
```


```python
#       Number of people vaccinated
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (12,24))
ax1.set_title("People Vaccinated Over Time (At Least 1 Shot)", fontsize = 22)
sns.lineplot(x = 'date', y = 'vaccinated', data = df, ax = ax1, hue = 'location')

ax1.legend()

ax1.set_xlabel('', fontsize = 18)
ax1.set_ylabel('Percentage of Population', fontsize = 18)

#       Number of people fully vaccinated
ax2.set_title("People Fully Vaccinated Over Time (2 Shots)", fontsize = 22)
sns.lineplot(x = 'date', y = 'fully_vaccinated', data = df, ax = ax2, hue = 'location')

ax2.legend()

ax2.set_xlabel('', fontsize = 18)
ax2.set_ylabel('Percentage of Population', fontsize = 18)

#       Number of people booster vaccinated
ax3.set_title("People Who Received Booster Shot Over Time (3 Shots)", fontsize = 22)
sns.lineplot(x = 'date', y = 'boostered', data = df, ax = ax3, hue = 'location')

ax3.legend()

ax3.set_xlabel('Date', fontsize = 18)
ax3.set_ylabel('Percentage of Population', fontsize = 18)

plt.tight_layout()
plt.show()  
```


    
![png](/images/covid_around_world/output_26_0.png)
    


Out of the countries on the list, we see the Israel was the first country to start widespread Covid-19 vaccinations, followed by United States and United Kingdom. However, in terms of percentage of population being vaccinated we observe that these three countries have been overtakened by many other countries since the third quarter of 2021.

In terms of booster shots, we see that only Israel has a large proportion of the population having taken the third shot.

When looking at vaccination rates an example of a common misconception is that going from 50% -> 60% vaccinated rate is the same as going from 60% -> 70% vaccinated rate. In both cases a 10% increase is seen in the vaccination rates. In reality, going from 50% -> 60% results in a 20% decrease in number of unvaccinated individuals, while going from 60% -> 70% results in a 25% decrease in the number of unvaccinated individuals. As the vaccination rate increases, each individual absolute percentage point increase results in a larger relative decrease in the unvaccinated population.

With the efficacy of vaccines significantly reducing fatality rates of infections, the effects of vaccines on Covid-19 death tolls will become more and more apparent as the vaccination rate increases.


```python
#       Current Percentages of Population that have received 0/1/2/3 vaccines
fig, ax = plt.subplots(figsize = (12,8))
ax.set_title("Percentage of Population Vaccinated", fontsize = 22)
sns.barplot(x = 'location', y = 'vaccinated', data = df.groupby('location')['vaccinated'].max().reset_index(), ax = ax, color = 'lightseagreen')
sns.barplot(x = 'location', y = 'fully_vaccinated', data = df.groupby('location')['fully_vaccinated'].max().reset_index(), ax = ax, color = 'teal')
sns.barplot(x = 'location', y = 'boostered', data = df.groupby('location')['boostered'].max().reset_index(), ax = ax, color = 'darkslategrey')


handles, labels = ax.get_legend_handles_labels()
patch = mpatches.Patch(color = 'lightseagreen', label = 'At Least Partially Vaccinated')
patch1 = mpatches.Patch(color = 'teal', label = 'Fully Vaccinated')
patch2 = mpatches.Patch(color = 'darkslategrey', label = 'Booster Vaccinated')
handles.append(patch)
handles.append(patch1)
handles.append(patch2)
ax.legend(handles = handles, loc = 'upper right')

ax.set_xlabel('Country', fontsize = 18)
ax.set_ylabel('Percentage of Population', fontsize = 18)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show() 
```


    
![png](/images/covid_around_world/output_28_0.png)
    




<a id = '5.4'></a>
### 5.4 Testings
In this section we shall look at data related to Covid-19 testing from the different countries. When discussing the pandemic usually the number of cases / deaths / vaccinations are used. However, one can argue that without knowing the extent of testing a proper picture of the pandemic cannot be obtained.


```python
## Testing  <--- A commonly missed data when discussing the pandemic (cases / deaths / vaccinations are the usual data)
q = """SELECT tests.*, others.population, others.gdp_per_capita, others.hospital_beds_per_thousand
FROM tests
JOIN others
ON tests.iso_code = others.iso_code
WHERE tests.iso_code IN ('USA','GBR','FRA','ITA','CAN','BRA','BEL','ISR',
                       'NLD','TWN','KOR','SGP','PHL','IND','AUS','PNG')"""
colnames =  column_names(q)

df = pd.DataFrame(data = query(q), columns = colnames)
```


```python
#       Number of tests 
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize = (12,24))
ax1.set_title("7 Day Average Tests Per Thousand", fontsize = 22)
sns.lineplot(x = 'date', y = 'new_tests_smoothed_per_thousand', data = df, ax = ax1, hue = 'location')

ax1.legend(loc = 'upper left')
ax1.set_xlabel('')
ax1.set_ylabel('Tests per Thousand People', fontsize = 18)

#       Number of tests per case
ax2.set_title("7 Day Average Tests Per Case", fontsize = 22)
sns.lineplot(x = 'date', y = 'tests_per_case', data = df, ax = ax2, hue = 'location')

ax2.legend(loc = 'upper left')
ax2.set_xlabel('')
ax2.set_ylabel('Tests per Confirmed Case', fontsize = 18)

#       Positive Rate
ax3.set_title("7 Day Average Positive Rate", fontsize = 22)
sns.lineplot(x = 'date', y = 'positive_rate', data = df, ax = ax3, hue = 'location')

ax3.legend(loc = 'upper left')
ax3.set_xlabel('Date', fontsize = 18)
ax3.set_ylabel('Tests Positive Rate', fontsize = 18)

plt.tight_layout()
plt.show() 
```


    
![png](/images/covid_around_world/output_31_0.png)
    


Looking at the average tests per case, we see that countries like Australia and Singapore have periods of time where there were a few thousand tests per confirmed case of Covid-19. During this period of time a more complete picture of the spread of the virus can be drawn as the high number of tests per case will suggest that less cases are escaping detections.

Alternatively, a country can use the positive rate as a gauge of whether their testing is adequate. A lower positive rate may suggest that less cases are escaping detections giving a clearer picture of the pandemic in a given country.



```python
#       Total number of tests (number of hospital beds) vs GDP per capita
q = """SELECT tests.iso_code, tests.location, tests.total_tests_per_thousand, 
others.population, others.gdp_per_capita, others.hospital_beds_per_thousand
FROM tests
JOIN others
ON tests.iso_code = others.iso_code"""
colnames =  column_names(q)

df = pd.DataFrame(data = query(q), columns = colnames)
```


```python
# Total Tests per Thousand vs GDP per Capita
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12,16))
ax1.set_title("Number of Total Tests per Thousand vs GDP per Capita", fontsize = 22)
sns.regplot(x = 'gdp_per_capita', y = 'total_tests_per_thousand', 
                data = df.groupby('location')['total_tests_per_thousand', 'gdp_per_capita'].max().reset_index(), ax = ax1,
                scatter_kws={"color": "teal"}, line_kws={"color": "salmon"})
ax1.set_xlabel('')
ax1.set_ylabel('Total Tests per Thousand People', fontsize = 18)
ax1.set_yscale('log')

# Total Hospital Beds per Thousand vs GDP per Capita
ax2.set_title("Number of Hospital Beds per Thousand vs GDP per Capita", fontsize = 22)
sns.regplot(x = 'gdp_per_capita', y = 'hospital_beds_per_thousand', 
                data = df.groupby('location')['hospital_beds_per_thousand', 'gdp_per_capita'].max().reset_index(), ax = ax2,
                scatter_kws={"color": "slategrey"}, line_kws={"color": "orange"})
ax2.set_xlabel('GDP per Capita (USD)')
ax2.set_ylabel('Hospital Beds per Thousand People', fontsize = 18)

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_34_0.png)
    


When looking at the total tests and total hospital beds per thousand people vs GDP per capita of the countries available in the dataset, positive correlations are found. This makes sense as it suggests that countries who are doing well economically have the resources to conduct more tests and procure more hospital beds.

<a id = '5.5'></a>
### 5.5 Excess Mortalities
In this section we shall look at excess mortalities across the countries.


```python
## Excess Mortality
q = """SELECT* FROM excess_mortality
WHERE excess_mortality.iso_code IN ('USA','GBR','FRA','ITA','CAN','BRA','BEL','ISR',
                       'NLD','TWN','KOR','SGP','PHL','IND','AUS','PNG')"""
colnames = column_names(q)

df = pd.DataFrame(data = query(q), columns = colnames)
df.date = pd.to_datetime(df.date)
```


```python
#       Weekly Excess Mortality
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (12,24))
ax1.set_title("Weekly Excess Mortality")
sns.lineplot(x = 'date', y = 'excess_mortality', data = df, hue = 'location', ax = ax1)
ax1.set_xlabel('')
ax1.set_ylabel('Weekly Excess Mortality (%)', fontsize = 18)

#       Cumulative Excess Mortality
ax2.set_title("Cumulative Excess Mortality Since Jan 2020")
sns.lineplot(x = 'date', y = 'excess_mortality_cumulative', data = df, hue = 'location', ax = ax2)
ax2.set_xlabel('')
ax2.set_ylabel('Cumulative Excess Mortality (%)', fontsize = 18)

#       Absolute Cumulative Excess Mortality
ax3.set_title("Cumulative Absolute Excess Mortality Since Jan 2020")
sns.lineplot(x = 'date', y = 'excess_mortality_cumulative_absolute', data = df, hue = 'location', ax = ax3)
ax3.set_xlabel('Date', fontsize = 18)
ax3.set_ylabel('Cumulative Excess Mortality (Number of People)', fontsize = 18)

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_37_0.png)
    


Looking at weekly excess mortalities, the graph closely mirrors the graph of daily new cases. Cumulative excess mortality suggests that many countries see excess mortalities of 5% to 15% within this period of the pandemic.

On a surface level 15% may not seem like a very large number, but converted to raw number of people we see some countries have seen almost a million excess deaths than usual over the past year and a half.

<a id = '5.6'></a>
### 5.6 Hospitalisations



```python
## Hospitalisation
q = """SELECT* FROM hospitalisation
WHERE hospitalisation.iso_code IN ('USA','GBR','FRA','ITA','CAN','BRA','BEL','ISR',
                       'NLD','TWN','KOR','SGP','PHL','IND','AUS','PNG')"""
colnames = column_names(q)

df = pd.DataFrame(data = query(q), columns = colnames)
df.date = pd.to_datetime(df.date)
```


```python
#       Number of hosp / weekly hosp
fig, (ax1,ax2) = plt.subplots(2,1, figsize = (12,16))

ax1.set_title('Number of Concurrent Hospital Admissions', fontsize = 22)
sns.lineplot(x = 'date', y = 'hosp_patients_per_million', data = df, ax = ax1, hue = 'location')
ax1.set_xlabel('Date', fontsize = 18)
ax1.set_ylabel('Concurrent Hospitalisations (Per Million)', fontsize = 18)

#       Number of ICU / weekly ICU
ax2.set_title('Number of Concurrent ICU Admissions', fontsize = 22)
sns.lineplot(x = 'date', y = 'icu_patients_per_million', data = df, ax = ax2, hue = 'location')
ax2.set_xlabel('Date', fontsize = 18)
ax2.set_ylabel('Concurrent Hospitalisations (Per Million)', fontsize = 18)

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_40_0.png)
    




<a id = '5.7'></a>
### 5.7 Other Data
Some other data available in the data set is stringency index of a country and some other miscellaneous stats such as median age, life expectancy, etc.

Stringency index is a metric ranging from 0 - 100 calculated from the various measures a country has applied to combat the spread of the virus with 100 being the more stringent measures.


```python
## Others / Stringency
q = """SELECT stringency.*, cases.new_cases_smoothed_per_million,
cases.new_deaths_smoothed_per_million
FROM stringency
JOIN cases
ON stringency.id = cases.id
WHERE stringency.iso_code IN ('USA','GBR','FRA','ITA','CAN','BRA','BEL','ISR',
                       'NLD','TWN','KOR','SGP','PHL','IND','AUS','PNG')
"""
colnames = column_names(q)

df = pd.DataFrame(data = query(q), columns = colnames)
df.date = pd.to_datetime(df.date)
```


```python
#       Cases / Deaths vs Stringency Index
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,16))
ax1.set_title('7 Day Average Cases vs Stringency Index', fontsize = 22)
sns.scatterplot(x = 'stringency_index', y = 'new_cases_smoothed_per_million', data = df.dropna(), hue = 'location', ax = ax1)
sns.regplot(x = 'stringency_index', y = 'new_cases_smoothed_per_million', data = df.dropna(), scatter = False, ax = ax1)
ax1.set_yscale('log')

ax2.set_title('7 Day Average Deaths vs Stringency Index', fontsize = 22)
sns.scatterplot(x = 'stringency_index', y = 'new_deaths_smoothed_per_million', data = df.dropna(), hue = 'location', ax = ax2)
sns.regplot(x = 'stringency_index', y = 'new_deaths_smoothed_per_million', data = df.dropna(), scatter = False, ax = ax2)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()
```


    
![png](/images/covid_around_world/output_43_0.png)
    


Looking at the number of cases/deaths we see that both are positively correlated to the stringency index. In particular, stringency index is more readily affected by the number of deaths suggesting that countries are more willing to apply stricter measures when faced with higher death tolls as compared to higher confirmed infections.


```python
#       Correlations between columns and deaths
q = """SELECT others.*, cases.total_cases_per_million,
cases.total_deaths_per_million
FROM others
JOIN cases
ON others.iso_code = cases.iso_code
"""

colnames = column_names(q)
df = pd.DataFrame(data = query(q), columns = colnames)

data = df.groupby('location')['population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita',  'cardiovasc_death_rate', 'diabetes_prevalence',
       'female_smokers', 'male_smokers', 
       'hospital_beds_per_thousand', 'life_expectancy',
       'human_development_index', 'total_cases_per_million',
       'total_deaths_per_million'].max()
```


```python
#       Correlation Matrix
fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(data = data.corr(), ax = ax, cmap = 'mako', annot = True)
ax.set_title("Correlation")
```




    Text(0.5, 1.0, 'Correlation')




    
![png](/images/covid_around_world/output_46_1.png)
    


On the surface level, we see many factors correlated with higher number of cases/deaths per million such as median age of population, life expectancy, etc.

For factors such as share of the population that is aged above 65/70, one can hypothesis that a larger proportion of aged population means that there may be more cross-generational social contact (e.g. grandparents and grandchild) leading to more rampant spread of the virus across different segments of the population. This can be verified by comparing the contact tracing data of countries with a higher median age vs lower median age..

Other factor such as GDP per capita sees a stronger correlation to total cases vs total deaths per million. This may be due to wealthier nations having the resources to conduct more widespread testing leading to less cases going undetected. Furthermore, wealthier countries may also be better equipped to combat infections leading to higher survival rates.

***

<a id = '6'></a>
## 6. Conclusion
In this notebook we have explored and discussed Covid-19 related data from around the world. Looking at some metrics commonly seen on mainstream media and social media and discussing their limitations and usefulness.
