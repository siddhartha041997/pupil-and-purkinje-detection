import matplotlib.pyplot as plt
import csv
import pandas as pd
import math
 
x=[]
y=[]
#a=[]
#b=[]
 
dataframe  = pd.read_csv('C:\\Users\\Envy\\Desktop\\anglenew6.csv')
x = dataframe.distance
y = dataframe.checker
#a = dataframe.num
#b = dataframe.distance
plt.scatter(x,y)
#plt.scatter(a,b)  
plt.title("scatter plot for angle and distance for subject 6")
plt.xlabel("distance")
plt.ylabel("angle")
plt.show() 

