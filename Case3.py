#!/usr/bin/env python
# coding: utf-8

# # DATASET LAADPAALDATA

# ### IMPORTING AND CLEANING DATA

# In[1]:

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# In[2]:


laadpaaldata = pd.read_csv('laadpaaldata.csv')
pd.set_option('display.max_columns', None)


# In[3]:


laadpaaldata.head()


# In[4]:


laadpaaldata.info()


# In[5]:


laadpaaldata.describe()


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.hist(laadpaaldata['ConnectedTime'], bins = 50)


# In[8]:


plt.hist(laadpaaldata['TotalEnergy'], bins=50)


# In[9]:


plt.hist(laadpaaldata['ChargeTime'], bins=50)


# In[10]:


plt.hist(laadpaaldata['MaxPower'], bins=50)


# In[11]:


laadpaaldata1 = laadpaaldata[laadpaaldata['ChargeTime'] >= 0 ]


# In[12]:


laadpaaldata1.describe()


# In[13]:


plt.hist(laadpaaldata1['ChargeTime'], bins=50)


# In[14]:


laadpaaldata1 = laadpaaldata1[laadpaaldata1['ConnectedTime']<=50]


# In[15]:


laadpaaldata1 = laadpaaldata1[laadpaaldata1['ChargeTime']<=15]


# In[16]:


laadpaaldata1['Percentage opladen'] = laadpaaldata1['ChargeTime'] / laadpaaldata1['ConnectedTime']


# In[17]:


laadpaaldata1.head()


# In[18]:


laadpaaldata1.info()


# In[19]:


plt.scatter(y=laadpaaldata1['TotalEnergy'], x=laadpaaldata1['ChargeTime'])


# In[ ]:





# In[ ]:





# DATASET laadpaaldata1 is klaar voor gebruik, volledig clean

# In[20]:


import plotly.express as px
import plotly.graph_objects as go


# In[21]:


medianConTime = laadpaaldata1['ConnectedTime'].median()
medianChaTime = laadpaaldata1['ChargeTime'].median()
meanConTime = laadpaaldata1['ConnectedTime'].mean()
meanChaTime = laadpaaldata1['ChargeTime'].mean()


# In[22]:


fig = go.Figure()
for col in ['ConnectedTime', 'ChargeTime']:
    fig.add_trace(go.Histogram(x=laadpaaldata1[col]))


dropdown_buttons = [
    {'label': 'Connected Time', 'method': 'update',
    'args': [{'visible': [True, False]},
            {'title': 'Connected Time'}]},
    {'label': 'Charge Time', 'method': 'update',
    'args': [{'visible': [False, True]},
            {'title': 'Charge Time'}]}]

float_annotation = {'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.8,'showarrow': False,
                    'text': f'Median of Connecting Time is {medianConTime} hours',
                    'font' : {'size': 15,'color': 'black'}
                    }


st.plotly_graph(fig.data[1].visible=False)
st.plotly_graph(fig.update_layout({'updatemenus':[{'type': "dropdown",'x': 1.3,'y': 0.5,'showactive': True,'active': 0,'buttons': dropdown_buttons}]}))
st.plotly_graph(fig.update_layout(xaxis_title='Time in hour',
                  yaxis_title="Number of observations"))
fig.update_layout({'annotations': [float_annotation]})
fig.show()


# In[23]:


laadpaaldata1 = laadpaaldata1[laadpaaldata1['ConnectedTime']<=20]
laadpaaldata1 = laadpaaldata1[laadpaaldata1['ChargeTime']<=6]


# In[24]:


import plotly.figure_factory as ff

group_1 = laadpaaldata1['ConnectedTime']
group_2 = laadpaaldata1['ChargeTime']

hist_data = [group_1, group_2]
group_labels = ['Connected Time', 'Charge Time']

fig = ff.create_distplot(hist_data, group_labels, colors=['blue','red'])
fig.update_layout({'title': {'text':'Distplot of Charge and Connecting Time'},
                   'xaxis': {'title': {'text':'Time in hours'}}})
fig.show()


# In[25]:


laadpaaldata1['ConnectedTime'].mean()


# In[26]:


laadpaaldata1['ChargeTime'].mean()


# In[27]:


laadpaaldata1['Percentage opladen'].mean()


# - eventueel kleur aanpassen
# - toevoegen van gemiddelde, mediaan en kansdichtheid

# In[28]:


fig = go.Figure()
for col in ['ConnectedTime', 'ChargeTime']:
    fig.add_trace(go.Scatter(x=laadpaaldata1[col], y=laadpaaldata1['TotalEnergy'], mode='markers'))

my_buttons = [{'label': 'Connected Time', 'method': 'update',
    'args': [{'visible': [True, False, False]},
            {'title': 'Connected Time'}]},
    {'label': 'Charge Time', 'method': 'update',
    'args': [{'visible': [False, True, False]},
            {'title': 'Charge Time'}]},
    {'label': 'Combined', 'method': 'update',
    'args': [{'visible': [True, True, True]},
            {'title': 'Combined'}]}]

fig.update_layout({
    'updatemenus': [{
      'type':'buttons','direction': 'down',
      'x': 1.3,'y': 0.5,
      'showactive': True, 'active': 0,
      'buttons': my_buttons}]})    
fig.update_layout(xaxis_title='Time in hour',
                  yaxis_title="Total energy used in Wh")
fig.data[1].visible=False
fig.show()    


# - Legenda voor combined

# # DATASET API OPENCHARGEMAP

# ### IMPORTING AND CLEANING DATA

# In[29]:


import pandas as pd
import requests
import csv
import json
url = 'https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=1000&key=74e5c90d-3e4f-4bbe-b506-233af06f55ca'
r = requests.get(url)
datatxt = r.text
datajson = json.loads(datatxt)
print(datajson)


# In[30]:


df = pd.json_normalize(datajson)
df.head()


# In[31]:


df['AddressInfo.Country.Title'].unique()


# In[32]:


pd.set_option('max_columns', None)


# In[33]:


labels = ['UserComments', 'PercentageSimilarity','MediaItems','IsRecentlyVerified','DateLastVerified',
         'UUID','ParentChargePointID','DataProviderID','DataProvidersReference','OperatorID',
         'OperatorsReference','UsageTypeID','GeneralComments','DatePlanned','DateLastConfirmed','MetadataValues',
         'SubmissionStatusTypeID','DataProvider.WebsiteURL','DataProvider.Comments','DataProvider.DataProviderStatusType.IsProviderEnabled',
         'DataProvider.DataProviderStatusType.ID','DataProvider.DataProviderStatusType.Title',
         'DataProvider.IsRestrictedEdit','DataProvider.IsOpenDataLicensed','DataProvider.IsApprovedImport',
         'DataProvider.License','DataProvider.DateLastImported','DataProvider.ID','DataProvider.Title',
         'OperatorInfo.Comments','OperatorInfo.PhonePrimaryContact','OperatorInfo.PhoneSecondaryContact',
         'OperatorInfo.IsPrivateIndividual','OperatorInfo.AddressInfo','OperatorInfo.BookingURL',
         'OperatorInfo.ContactEmail','OperatorInfo.FaultReportEmail','OperatorInfo.IsRestrictedEdit',
         'UsageType','OperatorInfo','AddressInfo.DistanceUnit','AddressInfo.Distance','AddressInfo.AccessComments',
         'AddressInfo.ContactEmail','AddressInfo.ContactTelephone2','AddressInfo.ContactTelephone1',
         'OperatorInfo.WebsiteURL','OperatorInfo.ID','UsageType.ID','StatusType.IsUserSelectable',
         'StatusType.ID','SubmissionStatus.IsLive','SubmissionStatus.ID','SubmissionStatus.Title',
         'AddressInfo.CountryID','AddressInfo.Country.ContinentCode','AddressInfo.Country.ID',
         'AddressInfo.Country.ISOCode','AddressInfo.RelatedURL','Connections']
df = df.drop(columns=labels)


# In[34]:


df.head(30)


# In[35]:


df['NumberOfPoints'].sum()


# In[36]:


df['OperatorInfo.Title'].unique()


# In[37]:


df['UsageCost'].unique()


# In[38]:


mappings = {'free':'Free',  '':'Unknown', 'Paod':'Paid','unknown':'Unknown','free at the bicycle chargeplace':'Free',
           'Gratis':'Free', 'gratis':'Free'}
df['UsageCost1'] = df['UsageCost'].replace(mappings)


# In[39]:


fig = go.Figure()
for col in ['OperatorInfo.Title', 'UsageCost1']:
    fig.add_trace(go.Histogram(x=df[col]))


dropdown_buttons = [
    {'label': 'Operator', 'method': 'update',
    'args': [{'visible': [True, False]},
            {'title': 'Operator of Charging Station'}]},
    {'label': 'Usage Cost', 'method': 'update',
    'args': [{'visible': [False, True]},
            {'title': 'Usage Cost of Charging Station'}]}]


fig.data[1].visible=False
fig.update_layout({'updatemenus':[{'type': "dropdown",'x': 1.3,'y': 0.5,'showactive': True,'active': 0,'buttons': dropdown_buttons}]})
fig.update_xaxes(tickangle = -45)
fig.show()


# In[40]:


df['AddressInfo.StateOrProvince'].unique()


# In[41]:


mapping = {'South Holland': 'Zuid-Holland', 'None': 'Nan','North Brabant':'Noord-Brabant',
          'North Holland': 'Noord-Holland', 'NH': 'Noord-Holland', 'North-Holland':'Noord-Holland',
          'UT':'Utrecht', 'Holandia Północna':'Noord-Holland', 'Seeland':'Zeeland', 'ZH':'Zuid-Holland',
          'Nordholland':'Noord-Holland','Stellendam':'Zuid-Holland', '':'Nan', 'FRL':'Friesland','Noord Holand':'Noord-Holland',
          'Noord Brabant':'Noord-Brabant', 'Stadsregio Arnhem Nijmegen':'Gelderland', 'Zuid-Holland':'Zuid-Holland'}


# In[42]:


df['Provincie'] = df['AddressInfo.StateOrProvince'].replace(mapping)


# In[43]:


fig = px.histogram(df, x='Provincie', 
                   title='Number of charging stations per province',
                  labels=dict(x='Province')).update_xaxes(categoryorder='total descending')
fig.show()


# In[ ]:





# In[44]:


df['LAT'] = df['AddressInfo.Latitude']
df['LNG'] = df['AddressInfo.Longitude']


# In[45]:


import folium


# In[46]:


m = folium.Map(location = [52.0893191, 5.1101691], 
               zoom_start = 7)

for row in df.iterrows():
    row_values = row[1]
    location = [row_values['LAT'], row_values['LNG']]
    marker = folium.Marker(location = location,
                         popup = row_values['AddressInfo.AddressLine1'])
    marker.add_to(m)

m


# In[47]:


import geopandas as gpd
provincies = gpd.read_file('provinces.geojson')


# In[48]:


provincies.head(20)


# In[49]:


m = folium.Map(location = [52.0893191, 5.1101691], 
               zoom_start = 7)
m.choropleth(
    geo_data = provincies,
    name = 'geometry',
    data = provincies)


# # DATASET ELEKTRISCHE VOERTUIGEN

# ### IMPORTING AND CLEANING DATA

# In[103]:


Elektrisch = pd.read_csv('Elektrische_voertuigen.csv')


# In[104]:


Elektrisch.head()


# In[105]:


Elektrisch.info()


# In[106]:


Elektrisch.describe()


# In[107]:


plt.hist(Elektrisch['Massa rijklaar'], bins = 40)


# In[108]:


plt.scatter(y=Elektrisch['Massa rijklaar'], x=Elektrisch['Massa ledig voertuig'])


# In[109]:


Elektrisch1 = Elektrisch[Elektrisch['Massa rijklaar'] > 750]
Elektrisch1['Massa rijklaar'].hist(bins = 40)
plt.show()


# In[110]:


pd.isna(Elektrisch1['Catalogusprijs']).sum()


# In[111]:


Elektrisch1['Catalogusprijs'].fillna(Elektrisch['Catalogusprijs'].mean(), inplace=True)


# #### SELECTION OF COLUMNS TO USE

# In[112]:


data = ['Kenteken','Merk', 'Handelsbenaming', 'Inrichting', 'Eerste kleur', 'Massa rijklaar', 'Zuinigheidslabel', 'Catalogusprijs'] 
df = Elektrisch1[data]


# In[113]:


df.head()


# In[114]:


pd.isna(df['Catalogusprijs']).sum()


# In[115]:


df.info()


# In[116]:


df['Zuinigheidslabel'].fillna(('Onbekend'), inplace=True)


# In[117]:


df['Zuinigheidslabel'].value_counts().sort_values()


# In[118]:


del df['Zuinigheidslabel']


# In[119]:


df.info()


# In[120]:


df.describe()


# In[121]:


df['Catalogusprijs'].max()


# In[122]:


df1 = df[df['Catalogusprijs'] <= 200000]
df1.info()


# In[123]:


plt.hist(df1['Catalogusprijs'], bins=100)


# In[124]:


df1['Eerste kleur'].value_counts()


# In[125]:


df1['Inrichting'].value_counts()


# In[126]:


df1.groupby("Merk")['Handelsbenaming'].unique()


# In[ ]:





# In[127]:


df1["Merk"].unique()


# In[128]:


mappings1 = {'TESLA MOTORS':'TESLA', 'BMW I': 'BMW', 'FORD-CNG-TECHNIK':'FORD', 'VW':'VOLKSWAGEN', 'VOLKSWAGEN/ZIMNY':'VOLKSWAGEN',
            'JAGUAR CARS': 'JAGUAR', 'ZIE BIJZONDERHEDEN':'Nan', 'VW-PORSCHE':'VOLKSWAGEN'}
df1["CarBrand"] = df1['Merk'].replace(mappings1)


# In[129]:


df1['CarBrand'].unique()


# In[130]:


fig = px.histogram(df1, x='CarBrand', 
                   title='Number of cars per brand',
                   labels={'CarBrand':'Brand of the car'}).update_xaxes(categoryorder='total descending')
fig.show()


# In[131]:


EV = pd.read_csv("EV_vanaf_2009.csv")


# In[132]:


EV.head()


# In[133]:


EV = EV.assign(Datum = pd.to_datetime(EV['Datum tenaamstelling'], format='%Y%m%d'))
EV['Datum'].head


# In[134]:


behouden = ['Kenteken', 'Datum']
newdf = EV[behouden]
newdf.head()


# In[135]:


mergeddf = pd.merge(df1, newdf, on="Kenteken")
mergeddf.head()


# In[136]:


del mergeddf['Merk']


# In[137]:


mergeddf.info()


# In[90]:


fig, ax = plt.subplots()
ax.plot(mergeddf['Datum'], mergeddf['Massa rijklaar'])
ax.set_xlabel('Time')
ax.set_ylabel('Number of Electric cars sold')
plt.show()


# In[139]:


mergeddf[mergeddf['CarBrand'] == 'TESLA'].value_counts('Handelsbenaming').unique


# In[140]:


mappings2 = {"TESLA MODEL 3":"MODEL 3", "MODEL S 70":"MODEL S", "MODEL S 85":"MODEL S",
            "MODEL S P85+":"MODEL S", "MODEL3":"MODEL 3", "S 75 D":"MODEL S", "TESLA MODEL S":"MODEL S"}
mergeddf['Type'] = mergeddf['Handelsbenaming'].replace(mappings2)
mergeddf.head()


# In[141]:


del mergeddf['Handelsbenaming']


# In[142]:


mergeddf.head()


# # MERGEDDF IS DATASET DIE KLAAR IS EN SCHOON

# In[143]:


Tesla = mergeddf[mergeddf['CarBrand']=='TESLA']
Tesla.head()


# In[145]:


fig = px.histogram(Tesla, x='Type', 
                   title='The different types of Tesla cars').update_xaxes(categoryorder='total descending')
fig.show()


# In[ ]:





# In[149]:


color_map = {"MODEL 3" : 'rgb(53,201,132)', "MODEL S" : 'rgb(196,201,67)', "MODEL X" : 'rgb(149,81,202)', "MODEL Y" : 'rgb(140,71,150)',"ROADSTER" : 'rgb(201,90,84)'}
fig = px.box(data_frame=Tesla, x=Tesla['Type'], y='Catalogusprijs', 
             color='Type', 
             color_discrete_map=color_map, 
             category_orders={'Type':['MODEL 3', 'MODEL S', 'MODEL X', 'MODEL Y', 'ROADSTER']},
             labels={"Type":"Type"})
fig.update_xaxes(title_text = 'Type Tesla')
fig.update_yaxes(title_text = 'Price')
fig.update_layout(title_text = "Boxplots of price per type Tesla")
fig.update_traces(width=0.3)
    
fig.show()


# In[ ]:




