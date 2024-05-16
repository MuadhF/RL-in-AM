import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfcinv

# Function to calculate the integrand for an integral
def calc_int(t_prime, t, a, sigma, v, x, y, z, l=0, i=0, delta_t=0, xi=False):
    if xi == True:
        x_i = (l / 2) + (-1) ** (i - 1) * ((t_prime - (i - 1) * delta_t) * v - (l / 2))
        x = x_i
    term1 = (t - t_prime) ** (- 0.5)
    term2 = 2 * (a) * (t - t_prime) + sigma ** 2    
    term3 = np.exp(-((x - v * t_prime) ** 2 + y ** 2) / (4 * a * (t - t_prime) + 2 * sigma ** 2) - (z ** 2) / (4 * a * (t - t_prime)))
    
    integrand = (term1 / term2) * term3
    return integrand
    
# Function to calculate the temperature at a certain location and time
def calc_temp(T0, alpha, P, v, rho, cp, a, sigma, x, y, z, t):
    #print("Calculating temperature")

    integral, _ = quad(calc_int, 0, t, args = (t, a, sigma, v, x, y, z))
    #print("Integral is ", integral)
    #print("Non Integral is ", (alpha * P) / (np.pi * rho * cp * (4 * np.pi * a) ** 0.5))
    temp = T0 + (alpha * P) / (np.pi * rho * cp * (4 * np.pi * a) ** 0.5) * integral
    return temp

# Function to calculate the width of the meltpool at a certain location and time
def calc_width(Tm, T0, alpha, P, v, rho, cp, a, sigma, kappa, x, y, z, t):
    #print("Calculating Width at", Tm)
    y = 0.00001
    temp = Tm + 1
    increment = 0.00001
    
    while(temp > Tm):
        temp = calc_temp(T0, alpha, P, v, rho, cp, a, sigma, x, y, z, t)
        #print(f'Temp at y = {y} is {temp} ')
        if temp > Tm:
            y += increment
    width = (y - increment) * 2
    
    delta_H = (P * alpha) / (np.pi * np.sqrt(a * v * sigma ** 3))
    hs = (kappa * Tm) / a
    check = delta_H / hs
    if check > 25:
        width = width / (1 + 0.05 * (check - 25))  
    
    #print("Width")
    return width
    
    


# Function to calculate the accumulated temperature
def calc_delta_T(T0, alpha, P, v, rho, cp, a, sigma, l, h, delta_t, n, x, y, z, t):
    
    # Modified function to calculate temperature for the accumulated temperature
    integrals = []
    for i in np.arange(1, n + 1):
        y_i = (i - 1) * h
        
        integral, _ = quad(calc_int, (i - 1) * delta_t, i * delta_t, args = (t, a, sigma, v, x, y_i, z, l, \
                                                                            i, delta_t, True))
        integrals.append(integral)
    sum_integrals = sum(integrals)
    #print(integrals)
    #print("Integral is ", integral)
    #print("Non Integral is ", (alpha * P) / (np.pi * rho * cp * (4 * np.pi * a) ** 0.5))
    
    delta_T = (alpha * P) / (np.pi * rho * cp * (np.pi * a) ** 0.5) * sum_integrals
    #print("Calculated delta T")
    return delta_T

# Calculating the temperature gradient
def calc_K(T0, Tm, Tb, v, a, sigma):
    #print("Calculating K")
    erftemp = (Tm - T0) / (Tb - T0)
    erf = erfcinv(erftemp) ** 2
    K = (Tb - T0) * np.exp(-erf) * ((np.sqrt(v)) / (np.sqrt(a * np.pi * sigma)))
    #print("Calculated K")
    return K

def calc_alpha(P, alpha_min, rho, cp, Tm, sigma, v):
    x = (P * alpha_min) / (np.pi * rho * cp * Tm * sigma **2 * v)
    alpha = 0.7 * (1 - np.exp(-0.66 * x))
    return alpha

# Calculating the melt pool depth
def calc_wd(T0, Tm, Tb, alpha, P, v, h, rho, cp, a, sigma, kappa, L):
    
    x_vals = np.arange(0.001,0.007,0.001)
    y = 0
    z = 0

    widths = []

    for x_val in x_vals:
        t = x_val / v
        #temperature = calc_temp(T0, alpha, P, v, rho, cp, a, sigma, x_val, y, z, t)
        #print(f'Temperature at {x_val}, {y}, {z} is {temperature}.\n')
        width = calc_width(Tm, T0, alpha, P, v, rho, cp, a, sigma, kappa, x_val, y, z, t)
        #print(f'Width at {x_val}, {y}, {z} is {width}')
        widths.append(width)

    width = sum(widths) / len(widths)
    #print("Width is", width)
    
    l = 0.006
    delta_t = l/v
    n = 20
    x = l / 2
    y = n * h
    z = 0
    t = n + 1/2

    delta_T = calc_delta_T(T0, alpha, P, v, rho, cp, a, sigma, l, h, delta_t, n, x, y, z, t)
    K = calc_K(T0, Tm, Tb, v, a, sigma)
    
    #print("Calculating Depth")
    
    numerator = (P * alpha) - (2 * np.pi / 3) * K * kappa * sigma * width
    denominator = (np.pi / 3) * K * kappa * (4 * sigma + width / 2) + (np.pi / 4) * v * rho * (L + cp * delta_T) * width
    depth = numerator / denominator
    #print("Depth is", depth)
    return width, depth


def calc_pi(P, v, h, thickness, T0, Tm, Tb, alpha_min, rho, cp, a, sigma, kappa, L, tens, visc):
    alpha = calc_alpha(P, alpha_min, rho, cp, Tm, sigma, v)
    #print("Calculating mp geometry")
    width, depth = calc_wd(T0, Tm, Tb, alpha, P, v, h, rho, cp, a, sigma, kappa, L)
    #print("Calculated mp geometry")  
    if width != 0:
        pi1 = (np.pi * depth * width) / (2 * h * thickness)

        vx = (depth * width * (v ** 2)) / (4 * a * (1.5 * (depth + width / 2) - np.sqrt(depth * width / 2)))

        pi2 = (tens * depth) / (vx * visc * sigma)
    else:
        pi1, pi2 = 0, 0
    
    return pi1, pi2