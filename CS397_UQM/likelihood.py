#!/usr/bin/env python
#
import numpy as np
import pandas as pd

df = pd.read_csv('time-series-19-covid-combined.csv')
data = df[df['Country/Region']=='US']
data.reset_index(inplace=True)
data.drop(['index','Date','Country/Region','Province/State','Lat','Long'],axis=1,inplace=True)
data = np.array(data)
#data[0] = Confirmed
#data[1] = Recovered
#data[2] = Deaths

#Constant parameters
N = 329197954. #total population
epsilon = 1/5.2 #transfer rate from latent to infectious
noise_sig = 500. #model noise

def model(q,beta,k,c1,c2,c3,deq,deqq,diq,delta,gamma,E0,I0):
    SEIeQRD = np.zeros((data.shape[0],7))
    SEIeQRD[0,0] = N - E0 - I0 - data[0,0] 
    SEIeQRD[0,1] = E0
    SEIeQRD[0,2] = I0
    SEIeQRD[0,4] = data[0,0]
    
    for day in range(1,data.shape[0]):
        SEIeQRD[day,0] = SEIeQRD[day-1,0] - beta*(c1+c2*np.exp(-c3*day))*SEIeQRD[day-1,0]*(k*SEIeQRD[day-1,1]+SEIeQRD[day-1,2])/N
        SEIeQRD[day,1] = SEIeQRD[day-1,1] + (1-q)*beta*(c1+c2*np.exp(-c3*day))*SEIeQRD[day-1,0]*(k*SEIeQRD[day-1,1]+SEIeQRD[day-1,2])/N \
                                          - epsilon*SEIeQRD[day-1,1] \
                                          - deq*SEIeQRD[day-1,1]
        SEIeQRD[day,2] = SEIeQRD[day-1,2] + epsilon*SEIeQRD[day-1,1] \
                                          - delta*SEIeQRD[day-1,2] \
                                          - diq*SEIeQRD[day-1,2]
        SEIeQRD[day,3] = SEIeQRD[day-1,3] + q*beta*(c1+c2*np.exp(-c3*day))*SEIeQRD[day-1,0]*(k*SEIeQRD[day-1,1]+SEIeQRD[day-1,2])/N \
                                          - deqq*SEIeQRD[day-1,3]
        SEIeQRD[day,4] = SEIeQRD[day-1,4] + deq*SEIeQRD[day-1,1] \
                                          + diq*SEIeQRD[day-1,2] \
                                          + deqq*SEIeQRD[day-1,3] \
                                          - delta*SEIeQRD[day-1,4] \
                                          - gamma*SEIeQRD[day-1,4]
        SEIeQRD[day,5] = SEIeQRD[day-1,5] + gamma*SEIeQRD[day-1,4]
        SEIeQRD[day,6] = SEIeQRD[day-1,6] + delta*SEIeQRD[day-1,2] \
                                          + delta*SEIeQRD[day-1,4]
    return SEIeQRD[:,4],SEIeQRD[:,5],SEIeQRD[:,6]

def likelihood(q,beta,k,c1,c2,c3,deq,deqq,diq,delta,gamma,E0,I0):
    Q, R, D = model(q,beta,k,c1,c2,c3,deq,deqq,diq,delta,gamma,E0,I0)
    loglike_Q = (-1. / (2. * noise_sig * noise_sig)) * np.dot(data[:,0] - Q,data[:,0] - Q)
    loglike_R = (-1. / (2. * noise_sig * noise_sig)) * np.dot(data[:,1] - R,data[:,1] - R)
    loglike_D = (-1. / (2. * noise_sig * noise_sig)) * np.dot(data[:,2] - D,data[:,2] - D)
    if np.isnan(loglike_Q + loglike_R + loglike_D):
        return 1.e-8
    else:
        return loglike_Q + loglike_R + loglike_D