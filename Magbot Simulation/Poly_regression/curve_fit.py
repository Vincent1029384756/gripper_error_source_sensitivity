import numpy as np
import matplotlib.pyplot as plt
from regression import regre, polynomial, fit_curve

#theta2 values
theta2 = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])

#tau_int values
tau = np.array([-6.25E-05, -9.08E-05, -1.09E-04, -1.23E-04, -1.44E-04, -1.63E-04, -1.82E-04, -1.95E-04, -2.25E-04, -2.58E-04, -2.82E-04, -3.15E-04, -3.47E-04, -3.86E-04, -4.22E-04, -4.71E-04, -5.26E-04, -5.72E-04, -6.43E-04])

#order of polynomial
m = 3

#number of interpolation
ni =  1000

#interpolate datapoints
x_inter, y_inter, r2 = fit_curve(theta2, tau, m, ni)

print("Coefficient of determination: ", r2)

#plot regression
plt.figure(1)
plt.plot(theta2,tau, 'ro')
plt.plot(x_inter, y_inter, label= 'fitted curve')
plt.xlabel('Theta2 [deg]')
plt.ylabel('Tau_int [N*m]')
plt.legend()
plt.show()