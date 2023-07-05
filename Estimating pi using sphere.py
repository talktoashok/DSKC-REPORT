from mpl_toolkits import mplot3d
%matplotlib inline
import numpy as np
import random
import matplotlib.pyplot as plt
fig = plt.figure()
points_insidesphere=0          #number of points lying inside sphere
points_total=0                 #number of points lying inside sphere

#empty arrays for storing the values of the points inside or outside the sphere of unit radius 
x_inside=[]
y_inside=[] 
z_inside=[]
x_outside=[]
y_outside=[]
z_outside=[]
pi=[]
darts=[]
for i in range(1000):
    
    x_coordinate=random.uniform(-1,1)
    y_coordinate=random.uniform(-1,1)
    z_coordinate=random.uniform(-1,1)
    
    d=x_coordinate**2+y_coordinate**2+z_coordinate**2    #defining sphere
    if d<=1:
        points_insidesphere+=1
        x_inside.append(x_coordinate)
        y_inside.append(y_coordinate)
        z_inside.append(z_coordinate)
    else:
        x_outside.append(x_coordinate)
        y_outside.append(y_coordinate)
        z_outside.append(z_coordinate)
        
    points_total+=1                      #increments number of points outside the circle
    value_of_pi_1=6*(points_insidesphere/points_total)
    pi.append(value_of_pi_1)
    darts_number=i
    darts.append(darts_number)
    
ax = plt.axes(projection='3d')
ax.scatter3D(x_inside,y_inside,z_inside,c=z_inside,cmap='Greens',s=1)
ax.scatter3D(x_outside,y_outside,z_outside,c=z_outside,alpha=.1,s=1)
plt.savefig('p7.pdf')
plt.show()
value_of_pi=6*(points_insidesphere/points_total)
print("value of pi as calculated with monte carlo simulation is:","%.2f" %value_of_pi)
plt.plot(darts,pi)
plt.savefig('p8.pdf')
plt.show()
