import numpy as np
import random
import matplotlib.pyplot as plt
points_insidecircle=0          #number of points lying inside circle
points_total=0                 #number of points lying inside circle

#empty arrays for storing the values of the points inside or outside the circle of unit radius 
x_inside=[]
y_inside=[]                    
x_outside=[]
y_outside=[]
pi=[]
darts=[]
for i in range(100000):
    
    x_coordinate=random.uniform(0,1)
    y_coordinate=random.uniform(0,1)
    
    d=x_coordinate**2+y_coordinate**2    #defining circle
    if d<=1:
        points_insidecircle+=1
        x_inside.append(x_coordinate)
        y_inside.append(y_coordinate)
    else:
        x_outside.append(x_coordinate)
        y_outside.append(y_coordinate)
        
    points_total+=1                      #increments number of points outside the circle
    value_of_pi_1=4*(points_insidecircle/points_total)
    pi.append(value_of_pi_1)
    darts_number=i
    darts.append(darts_number)
plt.scatter(x_inside,y_inside,s=1)
plt.scatter(x_outside,y_outside,s=1,alpha=.4)
plt.savefig('p1.pdf')
plt.show()
value_of_pi=4*(points_insidecircle/points_total)
print("value of pi as calculated with monte carlo simulation is:","%.2f" %value_of_pi)
plt.plot(darts,pi)
plt.savefig('p2.pdf')
plt.show()
