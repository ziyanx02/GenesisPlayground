import math

P = 10.0
Q = 0.1  # process noise covariance
R = 5.0  # measurement noise covariance
P_ss = Q + math.sqrt(Q**2 + Q * R)
K_ss = P_ss / (P_ss + R)
print(K_ss)
