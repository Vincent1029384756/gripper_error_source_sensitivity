import numpy as np

def LPfilter(i, output):
#The function of the low pass filter is to take in the q data of the previous <= 10 trials and calculate the average

    #initilize the sum of the past filter outputs
    q_t = np.array([0.0, 0.0])

    #when it is still within the first 10 trials, we will take the average of all past trials
    if i < 9:
        for a in range(i+1):
            q_t += output[a, 1:3]
    
    #If number of trials is greater or equal to 10, we will only calculate the average of the last 10 qf
    else:
        for b in range(i-8, i+1):
            q_t += output[b, 1:3]

    #calculate the average:
    q_avg = q_t/10
    
    return q_avg