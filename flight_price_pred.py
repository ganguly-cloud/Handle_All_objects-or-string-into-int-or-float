# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_excel('Data_Train.xlsx')
#print len(train_df)  # 10683

test_df = pd.read_excel('Test_set.xlsx')
#print len(test_df)  # 2671

#print train_df.head()

'''
       Airline Date_of_Journey    Source  ... Total_Stops Additional_Info  Price
0       IndiGo      24/03/2019  Banglore  ...    non-stop         No info   3897
1    Air India       1/05/2019   Kolkata  ...     2 stops         No info   7662
2  Jet Airways       9/06/2019     Delhi  ...     2 stops         No info  13882
3       IndiGo      12/05/2019   Kolkata  ...      1 stop         No info   6218
4       IndiGo      01/03/2019  Banglore  ...      1 stop         No info  13302

[5 rows x 11 columns]'''
#print test_df.head()
'''
             Airline Date_of_Journey  ... Total_Stops              Additional_Info
0        Jet Airways       6/06/2019  ...      1 stop                      No info
1             IndiGo      12/05/2019  ...      1 stop                      No info
2        Jet Airways      21/05/2019  ...      1 stop  In-flight meal not included
3  Multiple carriers      21/05/2019  ...      1 stop                      No info
4           Air Asia      24/06/2019  ...    non-stop                      No info

[5 rows x 10 columns]'''

full_df= train_df.append(test_df,sort=False)
#print full_df.tail()
'''
                Airline Date_of_Journey  ... Additional_Info Price
2666          Air India       6/06/2019  ...         No info   NaN
2667             IndiGo      27/03/2019  ...         No info   NaN
2668        Jet Airways       6/03/2019  ...         No info   NaN
2669          Air India       6/03/2019  ...         No info   NaN
2670  Multiple carriers      15/06/2019  ...         No info   NaN

[5 rows x 11 columns]'''

#print full_df.dtypes
'''
Airline             object
Date_of_Journey     object
Source              object
Destination         object
Route               object
Dep_Time            object
Arrival_Time        object
Duration            object
Total_Stops         object
Additional_Info     object
Price              float64
dtype: object'''

'''      FEATURE ENGINEERING     '''
# Splitting the data_of_journey to Data,month and year
full_df['Date']=full_df['Date_of_Journey'].str.split('/').str[0]
full_df['Month']=full_df['Date_of_Journey'].str.split('/').str[1]
full_df['Year']=full_df['Date_of_Journey'].str.split('/').str[2]

#print full_df.dtypes
'''
Airline             object
Date_of_Journey     object
Source              object
Destination         object
Route               object
Dep_Time            object
Arrival_Time        object
Duration            object
Total_Stops         object
Additional_Info     object
Price              float64
Date                object
Month               object
Year                object
dtype: object'''
# Converting default string to int
full_df['Date']=full_df['Date'].astype(int)
full_df['Month']=full_df['Date'].astype(int)
full_df['Year']=full_df['Date'].astype(int)

#print full_df.dtypes

'''
Airline             object
Date_of_Journey     object
Source              object
Destination         object
Route               object
Dep_Time            object
Arrival_Time        object
Duration            object
Total_Stops         object
Additional_Info     object
Price              float64
Date                 int32
Month                int32
Year                 int32
dtype: object'''

# Now we can drop those Date_of_Journey column

full_df = full_df.drop(['Date_of_Journey'],axis=1)

full_df['Arrival_Time']=full_df['Arrival_Time'].str.split(' ').str[0]

#print full_df.isnull().sum()

'''
Airline               0
Source                0
Destination           0
Route                 1
Dep_Time              0
Arrival_Time          0
Duration              0
Total_Stops           1
Additional_Info       0
Price              2671
Date                  0
Month                 0
Year                  0
dtype: int64'''

#print full_df[full_df['Total_Stops'].isnull()]

# NaN is replacing with '1 stop'
full_df['Total_Stops']=full_df['Total_Stops'].fillna('1 stop')

# 'Non-stop' is replacing with '0 stop'
full_df['Total_Stops']=full_df['Total_Stops'].replace('non-stop','0 stop')

# Asusual 'Stop' is created as object type
full_df['Stop']= full_df['Total_Stops'].str.split(' ').str[0]

#converting this object to int


full_df['Stop']=full_df['Stop'].astype(int)

# Drop that 'Total_Stops' column

full_df=full_df.drop(['Total_Stops'],axis=1)

full_df['Arrival_Hour'] = full_df['Arrival_Time'] .str.split(':').str[0]
full_df['Arrival_Minute'] =full_df['Arrival_Time'] .str.split(':').str[1]

full_df['Arrival_Hour']=full_df['Arrival_Hour'].astype(int)
full_df['Arrival_Minute']=full_df['Arrival_Minute'].astype(int)
full_df=full_df.drop(['Arrival_Time'],axis=1)

full_df['Departure_Hour'] = full_df['Dep_Time'] .str.split(':').str[0]
full_df['Departure_Minute'] = full_df['Dep_Time'] .str.split(':').str[1]


full_df['Departure_Hour']=full_df['Departure_Hour'].astype(int)
full_df['Departure_Minute']=full_df['Departure_Minute'].astype(int)
full_df=full_df.drop(['Dep_Time'],axis=1)

full_df['Route_1']=full_df['Route'].str.split('→ ').str[0]
full_df['Route_2']=full_df['Route'].str.split('→ ').str[1]
full_df['Route_3']=full_df['Route'].str.split('→ ').str[2]
full_df['Route_4']=full_df['Route'].str.split('→ ').str[3]
full_df['Route_5']=full_df['Route'].str.split('→ ').str[4]

full_df['Price'].fillna((full_df['Price'].mean()),inplace=True)

full_df['Route_1'].fillna("None",inplace=True)
full_df['Route_2'].fillna("None",inplace=True)
full_df['Route_3'].fillna("None",inplace=True)
full_df['Route_4'].fillna("None",inplace=True)
full_df['Route_5'].fillna("None",inplace=True)

full_df=full_df.drop(['Route'],axis=1)
full_df=full_df.drop(['Duration'],axis=1)

#print full_df.isnull().sum()

'''
Airline             0
Source              0
Destination         0
Additional_Info     0
Price               0
Date                0
Month               0
Year                0
Stop                0
Arrival_Hour        0
Arrival_Minute      0
Departure_Hour      0
Departure_Minute    0
Route_1             0
Route_2             0
Route_3             0
Route_4             0
Route_5             0
dtype: int64'''


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() # to convert all object into int
full_df["Airline"]=encoder.fit_transform(full_df['Airline'])
full_df["Source"]=encoder.fit_transform(full_df['Source'])
full_df["Destination"]=encoder.fit_transform(full_df['Destination'])
full_df["Additional_Info"]=encoder.fit_transform(full_df['Additional_Info'])
full_df["Route_1"]=encoder.fit_transform(full_df['Route_1'])
full_df["Route_2"]=encoder.fit_transform(full_df['Route_2'])
full_df["Route_3"]=encoder.fit_transform(full_df['Route_3'])
full_df["Route_4"]=encoder.fit_transform(full_df['Route_4'])
full_df["Route_5"]=encoder.fit_transform(full_df['Route_5'])

''' Create Model '''

from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


# Deviding train & test data

train_df = full_df[0:10683]
test_df = full_df[10683:]

x = train_df.drop(['Price'],axis=1)
y = train_df['Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(x_train,y_train)
# It will get True which ever are the colums are support like required or not
#print model.get_support()
'''
[ True  True  True  True  True False False  True  True  True  True  True
  True False False False False]'''


selected_features=x_train.columns[(model.get_support())]
print selected_features
'''
Index([u'Airline', u'Source', u'Destination', u'Additional_Info', u'Date',
       u'Stop', u'Arrival_Hour', u'Arrival_Minute', u'Departure_Hour',
       u'Departure_Minute', u'Route_1'],
      dtype='object')'''

# so now we can drop year


x_train=x_train.drop(['Year'],axis=1)
x_test = x_test.drop(['Year'],axis=1)






              
