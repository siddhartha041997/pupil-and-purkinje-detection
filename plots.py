import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import datetime
from timeit import default_timer as timer

x=[]
y=[]
a=[]
b=[]
datet = str(datetime.datetime.now())

with open('C:\\Users\\Envy\\Desktop\\for thesis\\data_Sample_1.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter=',')
    for column in plots:
        x.append(float(column[6]))
        y.append(float(column[7]))
        a.append(float(column[8]))
        b.append(float(column[9]))
        vec1 = math.sqrt((float(column[6])*float(column[6])) + (float(column[7])*float(column[7])))
        vec2 = math.sqrt((float(column[8])*float(column[8])) + (float(column[9])*float(column[9])))
        disbetvec = math.sqrt(((float(column[6]) - float(column[8]))*(float(column[6]) - float(column[8]))) + ((float(column[7])-float(column[9]))*(float(column[7])-float(column[9]))))
        angle = ((vec1)*(vec1) + (vec2)*(vec2) - (disbetvec)*(disbetvec)) / (2*(vec1)*(vec2))
        print (angle)
        angleInDegree = math.degrees(math.acos(angle))
        print("θ =",angleInDegree,"°")
        distance = math.sqrt(((float(column[6])-float(column[8]))**2) + (float(column[7])-float(column[9]))**2)
        print ("Distance =",distance)
    
        with open('C:\\Users\\Envy\\Desktop\\t1.csv', 'a', newline='') as outfile:   
            fieldnames = ['time','angle','distance']
            output = csv.DictWriter(outfile, fieldnames=fieldnames)
            output.writeheader()
            output.writerow({ 'time':datet, 'angle':angleInDegree, 'distance':distance})
            output.writerow({})
            outfile.close()

#fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('plot for pupil and 1st purkinje coordinates')
#ax1.plot(x, y)
#ax2.plot(a, b)  
#plt.show()




