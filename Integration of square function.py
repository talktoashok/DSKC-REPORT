import numpy as np
import random
import matplotlib.pyplot as plt
points_insidecircle=0          #number of points lying downside the square curve
points_total=0                 #number of points lying upside the square curve

#empty arrays for storing the values of the points up or down the square curve
x_inside=[]
y_inside=[]                    
x_outside=[]
y_outside=[]
inte=[]
darts=[]
a=2
b=5
for i in range(50000):
    
    x_coordinate=random.uniform(2,5)
    y_coordinate=random.uniform(0,25)
    
    d=y_coordinate   
    if d<=x_coordinate**2:
        points_insidecircle+=1
        x_inside.append(x_coordinate)
        y_inside.append(y_coordinate)
    else:
        x_outside.append(x_coordinate)
        y_outside.append(y_coordinate)
        
    points_total+=1                      #increments number of points downside the curve
    int1=75*(points_insidecircle/points_total)
    inte.append(int1)
    darts_number=i
    darts.append(darts_number)
plt.scatter(x_inside,y_inside,s=1)
plt.scatter(x_outside,y_outside,s=1,alpha=.4)
plt.savefig('p8.pdf')
plt.show()
int1=75*(points_insidecircle/points_total)
print("Integration is : ","%.2f" %int1)
plt.plot(darts,inte)
plt.savefig('p9.pdf')
plt.show()
