import numpy as np
import random
import math
import matplotlib.pyplot as plt
points_insidecircle=0          #number of points lying downside the square curve
points_total=0                 #number of points lying upside the square curve

#empty arrays for storing the values of the points up or down the square curve
x_inside=[]
y_inside=[]                    
x_outside=[]
y_outside=[]
pi=[]
darts=[]
a=2
b=5


for i in range(500000):
    
    x_coordinate=random.uniform(0,6.28)
    y_coordinate=random.uniform(-1,1)
    
    d=math.sin(x_coordinate)
    if y_coordinate>=0:
        if d>=y_coordinate:
            points_insidecircle+=1
            x_inside.append(x_coordinate)
            y_inside.append(y_coordinate)
        else:
            x_outside.append(x_coordinate)
            y_outside.append(y_coordinate)
    else:
        if d<=y_coordinate:
            points_insidecircle+=1
            x_inside.append(x_coordinate)
            y_inside.append(y_coordinate)
        else:
            x_outside.append(x_coordinate)
            y_outside.append(y_coordinate)
        
        
    points_total+=1                      #increments number of points downside the curve
    value_of_pi_1=3.14*(points_insidecircle/points_total)
    pi.append(value_of_pi_1)
    darts_number=i
    darts.append(darts_number)
plt.scatter(x_inside,y_inside,s=1)
plt.scatter(x_outside,y_outside,s=1,alpha=.4)
plt.savefig('p11.pdf')
plt.show()
value_of_pi=6.28*(points_insidecircle/points_total)
print("Integration is : ","%.2f" %value_of_pi)
plt.plot(darts,pi)
plt.savefig('p12.pdf')
plt.show()
