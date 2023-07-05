## cost of pudding using Monte Carlo
import numpy as np 
import matplotlib.pyplot as plt 

n = 200000

m = np.random.uniform(18, 20, n)   
E = [2, 2.5, 3]
e = np.random.choice(E, n, [0.2, 0.5, 0.3])
s = np.random.uniform(20, 23, n)
b = np.random.normal(25, 1, n)

A = [500, 700]
a = np.random.choice(A, n, [0.7, 0.3])

P = np.zeros(n)  # Initialize P array

for i in range(n):
    P[i] = 2 * m[i] + 2 * e[i] + s[i] / 10 + 2 * b[i]/12 + a[i] / 10+10


print('Average price of the pudding to get profit of 10 ruppees over the year is : ', sum(P)/n)
plt.hist(P, bins=50)
plt.savefig('p10.pdf')


