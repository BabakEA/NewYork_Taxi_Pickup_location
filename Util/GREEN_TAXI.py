
import argparse
from IPython.display import clear_output
import sched, time
import pandas as pd
import numpy as np

import uszipcode

from datetime import *
import plotly.express as px

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler,StandardScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from ipywidgets import HBox, Layout, VBox , Label
import pprint
import ipywidgets as widgets

#import geopy
import time

from uszipcode import SearchEngine

search = SearchEngine()
import json

import haversine as hs #### distance calcualtore

import plotly.graph_objects as go
from plotly.subplots import make_subplots


mpl.rcParams["figure.figsize"]=(20,12)
mpl.rcParams['axes.grid']=True
pd.options.display.float_format = '{:.7f}'.format


import requests
from datetime import datetime

import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
pd.options.display.float_format = '{:.4f}'.format

Report=dict()


AREA_ZIP={"Manhattan":{"LAT":40.7831,"LNG":-73.9712},
         "Staten":{"LAT":40.5795,"LNG":-74.1502},
          "Bronx":{"LAT":40.8448,"LNG":-73.8648},
          "Queens":{"LAT":40.7282,"LNG":-73.7949},
          "Brooklyn":{"LAT":40.6782,"LNG":-73.9442}
}

def data_cleansing(df):
    
    print("Before cleansing shape : {}".format(df.shape))
    print("----- CLEANSING -----")
    

    
    df = df.drop(df[df['Pickup_longitude'] <= -74.5].index)
    df = df.drop(df[df['Pickup_longitude'] >= -73.5].index)
    print(df.shape)
   
    df = df.drop(df[df['Pickup_latitude'] <= 40.4].index)
    df = df.drop(df[df['Pickup_latitude'] >= 41].index)
    print(df.shape)
   
    df = df.drop(df[df['Dropoff_longitude'] <= -74.5].index)
    df = df.drop(df[df['Dropoff_longitude'] >= -73.5].index)
    print(df.shape)
   
    df = df.drop(df[df['Dropoff_latitude'] <= 40.4].index)
    df = df.drop(df[df['Dropoff_latitude'] >= 41].index)
    print(df.shape)
    
    #df=df.query("PICKUP_ZIP <= 13000 and PICKUP_ZIP >= 10000 ")
    #df=df.query("DROP_ZIP <= 13000 and DROP_ZIP >= 10000 ")
    print(df.shape)
    print("----- CLEANSING -----")
    print("After cleansing shape : {}".format(df.shape))
    
    return df



def ZIP_CODE(LAT,LNG):
    # return zipcode from lat and lang
    return search.by_coordinates(float(LAT), float(LNG), returns=1)[0].to_dict()["zipcode"]

def ZIP_to_points(zipcode):
    # return Zipcode based on lat and long
    test=search.by_zipcode(str(int(float(zipcode)))).to_dict()
    
    print(test["lat"],test["lng"])
    return test["lat"],test["lng"]

def Day_Hour_Quarter(x):
    DAY=x.day
    WeekDAY=x.strftime("%A")
    HOUR=x.hour
    MIN=int(x.minute /10)*10
    
    """
    print(test.day)
    print(test.strftime("%A"))
    print(test.hour)
    print(test.minute)
    """
    
    return DAY, WeekDAY,HOUR,MIN

def read_Json(path="./data/ZIP_CODE.json"):
    with open (path, "r") as f:

        # Reading from file
        ZIP_DICT = json.loads(f.read())
    return ZIP_DICT


def ZIP_Cluster(ZIPCOD):
    ZIP=str(int(float(ZIPCOD)))
    #### get zip cod: return Zipcod Area
    
    LAT, LNG=ZIP_to_points(ZIP)
    
    ####     {ZIP:{"ZONE":ZONE,"ZONE_ID":ZONE_ID,"AREA":AREA,"AREA_ID":AREA_ID}}


    return ZIP_DICT[ZIP]["ZONE"],ZIP_DICT[ZIP], LAT, LNG
    
    
def ConcatX(x:list):
    ## concat columns to gather
    return "_".join(str(item) for item in x)


def Float_INT(x,y,z):
    """#Fix format
    df["longitude"],df["latitude"],df["ZIP"]=zip(*df.apply(lambda x: Float_INT(x.longitude,x.latitude,x.ZIP),axis=1))

    """
    
    return int(x*1000), int(y*1000), int(z)



def basae_file(ZIP,DAY,HR):
    """
    creat emplty taxi location: calculate prob
    
    
    """
    data=[]
    for Zip in ZIP:
        for PDAY in DAY:
            for hr in HR:
                data.append([Zip,PDAY,hr,0])
                
    base_df=pd.DataFrame(columns=recommender_zip_2.columns.tolist(),data=data)
    return base_df


def distance_calculatore(A:list,B:list):
    A=[round(float(x),5)  for x in A]
    B=[round(float(x),5)  for x in B]
    
    
    return hs.haversine(A,B,unit=hs.Unit.METERS)
def mile_km(x):
    return float(x)*1.60934



def Newyour_ZIP(path="./data/New_your_zip.csv"):
    ZCD=pd.read_csv(path)
    ZCOD=dict(zip(ZCD["ZIPCODE"],ZCD["AREA"]))
    return ZCOD
def LAT_LNG(x):
    return AREA_ZIP[x]["LAT"],AREA_ZIP[x]["LNG"]

#### add Zoon Area to Dataset based on Postalcode

def ZIP_CLEANER(x):
    try:
        return NY_ZIP[x]
    except:
        return np.nan


############
#### Load data

#df=pd.read_csv("./updated_RAW_TAXI.csv")

ZIP_DIC=read_Json()

############### data Cleaning ###########
#df=data_cleansing(df)

############### Creat Newuyork Zip_code Dict ##########

NY_ZIP=Newyour_ZIP() ### get Zoone are for each postalcode

"""
############### Update Dtafram ##############
df["AREA"]=df.apply(lambda x: ZIP_CLEANER(x.PICKUP_ZIP) ,axis=1)
df=df.dropna()


A_df=df[["AREA" ,'Trip_distance', 'PICKUP_ZIP', 'P_Day','P_WEEK_Day', 'P_Hour', 'flag']]
A_df=A_df.groupby(["AREA","PICKUP_ZIP","P_Day","P_WEEK_Day","P_Hour"])['flag',"Trip_distance"].sum().reset_index()
### convert Mile to KM 
A_df["Trip_distance"]=df.apply(lambda x: mile_km(x.Trip_distance),axis=1)
A_df.head()

"""

################################### data visualization using plotly dashboard
############################################################################33

class Monthly_report: 
    def __init__(self,df):
        self.df=df
        #self.DICT=DICT
        self.MCC_LIST=df["AREA"].unique().tolist()
        self.BTN_select = widgets.Button(description='Analysis')
        self.BTN_outt = widgets.Output()
        
        def on_button_clicked(_):
            with self.BTN_outt:
                #clear_output()
                #print('Button clicked')
                self.BTN_outt.clear_output()
                widgets.Output().clear_output()
                #print("test")
                #print(self.Portfolio,)
                #print(self.Optimiser, self.Portfolio, self.MCC_Group,self.Best_Case_Slider.value)
                print("DAY : ",self.DAY_OF_MONTH," AREA : ", self.AREA," Hour : ", self.Daily_Hour)
                
                DD=str(self.DAY_OF_MONTH).zfill(2)
                HH=str(self.Daily_Hour).zfill(2)
                
                
                if self.AREA == "ALL":

                    tem=df.query("P_Day == '{DAY}' ".format(DAY=self.DAY_OF_MONTH))
                else:
                    tem=df.query("P_Day == '{DAY}' and AREA == '{area}'".format(DAY=self.DAY_OF_MONTH,area=self.AREA,))
                
                
                
                self.HR_lines=df.query("P_Day == '{DAY}' ".format(DAY=DD)
                                 ).groupby(["AREA","P_Hour"])['flag',"Trip_distance"].sum().reset_index()
                
                                
                #elf.HR_lines=self.HR_lines.sort_values(by=['P_Hour'])
                
                
                self.HR_lines["X"]=self.HR_lines.apply(lambda x: ConcatX(["2016_feb_{DAY}_Hour".format(DAY=DD)
                                                                ,x.P_Hour]),axis=1)
            
                self.HR_lines=self.HR_lines.pivot(index="X",columns="AREA",values="Trip_distance").fillna(0)
                
                
                
                
                self.HR_Days=df.query("P_Hour == '{Hour}' ".format(Hour= HH) 
                                ).groupby(["AREA","P_Day"])['flag',"Trip_distance"].sum().reset_index()
                
                
                
                #self.HR_Days=self.HR_Days.sort_values(by=['P_Day'])
                self.HR_Days["X"]=self.HR_Days.apply(lambda x: ConcatX(["2016_feb",x.P_Day,"Hour_{HR}".format(HR=HH)]
                                                             ),axis=1)
                
                
                
                self.HR_Days=self.HR_Days.pivot(index="X",columns="AREA",values="Trip_distance").fillna(0)
                
                

                self.Toal_trip=tem["flag"].sum()
                self.Toal_Disctance=tem["Trip_distance"].sum()
                
                self.MAX_trip=tem["flag"].max()
                self.MAX_Disctance=tem["Trip_distance"].max()
                
                self.AVG_trip=tem["flag"].mean()
                self.AVG_Disctance=tem["Trip_distance"].mean()
                
                self.total_dis=df.groupby(["AREA"])['flag',"Trip_distance"].sum().reset_index()
                
                
                self.plot()
                
                #tem=tem.replace(0, np.nan)
                """Dict_tem={}
                Dict_tem[self.Portfolio]={}
                Dict_tem[self.Portfolio][self.MCC_Group]={}
                Dict_tem[self.Portfolio][self.MCC_Group]=self.DICT[self.Portfolio][self.MCC_Group]
                pprint.pprint(Dict_tem)
                Dict_tem[self.Portfolio][self.MCC_Group]['BEST']['Recovery']=self.Best_Case_Slider.value
                Dict_tem[self.Portfolio][self.MCC_Group]['Months_from_30-April-2020']=self.Best_Case_Slider.value
                
                
                Dict_tem=Dat_picekr(Dict_tem, self.Best_Case_Slider.value, Portfolio =self.Portfolio ,
                                    MCC = self.MCC_Group, Start = "2020-03-01",method=0, ratio=self.Optimiser)
                """
                
                
                """tem= Recovery_simulatore(DICT,tem,Portfolio=self.Portfolio,MCC=self.MCC_Group,plot=0).df 
                Portfolio_Mcc_Plot(tem)"""
                
                
                #print(tem)
                
                
                


        def Chooser(DAY_OF_MONTH, AREA, Daily_Hour):
            HBOX_COMMENT=HBox([Label('GREEN TAXI Reporting System ....."')])

            self.BTN_select.on_click(on_button_clicked) #Function to run on click

            container2 = widgets.VBox([HBOX_COMMENT,self.BTN_select,self.BTN_outt])

            display(container2)
            self.OPT=DAY_OF_MONTH
            self.DAY_OF_MONTH=DAY_OF_MONTH
            self.AREA=AREA
            self.Daily_Hour=Daily_Hour

            #return (Optimiser,self.slider.value, Portfolio, MCC_Group)

        _ = widgets.interact(
                    Chooser, 
                    DAY_OF_MONTH=(1, 29, 1), 
                    AREA=["ALL",'Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten'], 
                    Daily_Hour=[x for x in range(24)],
                    #DICT=[DICT]
                    )
        
    def plot(self):
        Line_dict={"Brooklyn":{"color":'green', "width":2},
          "Bronx":{"color":'royalblue', "width":2, "dash":'dash'},
           "Manhattan":{"color":'orange', "width":2, "dash":'dot'},
           "Queens":{"color":'darkgray', "width":2, "dash":'dot'},
           "Staten":{"color":'yellowgreen', "width":2, }
          }

        fig_DAY_LIN = go.Figure()

        col=["Bronx","Brooklyn","Manhattan","Queens","Staten"] 
        fig_DAY_LIN = go.Figure()
        fig_HR_DYS = go.Figure()
        #for column in df.columns.to_list():
        for column in col:
            fig_DAY_LIN.add_trace(
                go.Scatter(
                    x = self.HR_lines.index,
                    y = self.HR_lines[column],
                    name = column,
                    line=Line_dict[column]
                )
            )
            
            ######## HR ######### 
            fig_HR_DYS.add_trace(
                go.Scatter(
                    x = self.HR_Days.index,
                    y = self.HR_Days[column],
                    name = column,
                    line=Line_dict[column]
                )
            )

        fig_DAY_LIN.update_layout(
        autosize=True,
        title="Hourly distance trip reported at {OO}:00 ".format(OO=str(self.DAY_OF_MONTH).zfill(2)),


        width=800,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        #paper_bgcolor="LightSteelBlue",
        paper_bgcolor="white",)
        ###############################################33
        fig_HR_DYS.update_layout(
        autosize=True,
        title="Daily distance trip reported on FEB {OO}".format(OO=str(self.Daily_Hour).zfill(2)),
        width=800,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        #paper_bgcolor="LightSteelBlue",
        paper_bgcolor="white",)

        
        """ 
        data = [fig_DAY_LIN, fig_HR_DYS]

        # Layout component
        layout = go.Layout(xaxis = dict(domain = [0.0, 0.45]),
                           xaxis2 = dict(domain = [0.55, 1.0]),
                           yaxis2 = dict(overlaying='y',
                                         anchor = 'free',
                                         position = 0.55
                                        )
                          )

        # Figure component
        fig = go.Figure(data=data, layout=layout)

        offline.iplot(fig)
        """

        fig_DAY_LIN.show()
        fig_HR_DYS.show()
        
        """
        self.Toal_trip=tem["flag"].sum()
        self.Toal_Disctance=tem["Trip_distance"].sum()

        self.MAX_trip=tem["flag"].max()
        self.MAX_Disctance=tem["Trip_distance"].max()

        self.AVG_trip=tem["flag"].mean()
        self.AVG_Disctance=tem["Trip_distance"].mean()
        """


        
        
        fig = make_subplots(
        rows = 13, cols = 6,

        specs=[[None, None, None,               None,None,None ],
               [None, None, None,               None,None,None ],
                [{"type": "indicator","rowspan": 2, "colspan": 2},None,None, 
                 {"type": "indicator","rowspan": 2, "colspan": 2},None,None],
                [None, None, None,               None,None,None],
                [None, None, None,               None,None,None],
                [None, None, None,               None,None,None, ],
                [{"type": "indicator","rowspan": 2, "colspan": 2},None,None,
                 {"type": "indicator","rowspan": 2, "colspan": 2},None,None],
                [    None, None, None,               None ,None,None],


               [None, None, None,               None,None,None ],
                [ {"type": "bar","rowspan": 2, "colspan": 2},None,None ,
                 {"type": "bar","rowspan": 2, "colspan": 2},None,None ],
                [None, None, None,               None,None,None ],
                [None, None, None,               None,None,None ],
               [None, None, None,               None,None,None]
              ])

        
        fig.add_trace(
           go.Indicator(mode="gauge+number", value=self.Toal_trip, 
              title={'text': "Total Trip on FEB {day} at {HR}:00 ".format(
                  day=self.Daily_Hour,HR=self.Daily_Hour)},number={"font":{"size":29}}),
            row=3, col=1)
        
        # traces with separate domains to form a subplot


        fig.add_trace( go.Indicator(mode="gauge+number", value=self.Toal_Disctance,
              title={'text': "Total Traveled Distance on FEB {day} at {HR}:00  KM".format(
                  day=self.Daily_Hour,HR=self.Daily_Hour)},number={"font":{"size":29}}),
                row=3, col=4)      

        
                # traces with separate domains to form a subplot
        fig.add_trace(go.Indicator(mode="gauge+number", value=self.MAX_trip, 
                title={'text': "Max Trip on FEB {day} at {HR}:00 ".format(
                    day=self.Daily_Hour,HR=self.Daily_Hour)},number={"font":{"size":29}}),
                     row=7, col=1) 

        fig.add_trace(go.Indicator(mode="gauge+number", value=self.MAX_Disctance,
                title={'text': "Max Traveled Distance on FEB {day} at {HR}:00  KM".format(
                    day=self.Daily_Hour,HR=self.Daily_Hour)},number={"font":{"size":29}})
                      ,row=7, col=4) 
        
       
        fig.add_trace(go.Bar(
            x=self.total_dis["AREA"].tolist(),
            y=self.total_dis["Trip_distance"],
            name='Total Monthly distances',
            marker_color='orange'

        ),row=10, col=1)
        
               
        fig.add_trace(go.Bar(
           
            x=self.total_dis["AREA"].tolist(),
            y=self.total_dis["flag"],
            
            name='Total Monthly trip',
            marker_color='green'

        ),row=10, col=4)

        
        # layout and figure production
        fig.update_layout(
        template="plotly_dark",
        title = "NewYork Green TAXI FEB 2016 ",
        font=dict(
        family="Courier New, monospace",
        size=15
        ),
        showlegend=True,
        legend_orientation="h",
        
       
        geo = dict(
                projection_type="orthographic",
                showcoastlines=True,
                landcolor="white", 
                showland= True,
                showocean = True,
                lakecolor="LightBlue"
        ),

        annotations=[
            dict(
                text="Source: https://www.linkedin.com/in/babak-emami/",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.35,
                y=-0.5)
        ])

        
        
        
        
        """layout = go.Layout(height = 800,
                           width = 900,
                           autosize = False,
                           title = 'Daily Analysis')"""
        #fig = go.Figure(data = [trace1, trace2,trace3, trace4,], layout = layout)
        fig.show()





            
            
            










