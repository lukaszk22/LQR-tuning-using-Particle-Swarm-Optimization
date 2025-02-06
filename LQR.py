import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Definition of tuning constants

# Zaproponowane wagi
q1 = 1000
q2 = 100
q3 = 100
q4 = 1
q5 = 5
q6 = 500000
q7 = 100
q8 = 100
q9 = 10
q10 = 1000
q11 = 1000
q12 = 10000

r1 = 1
r2 = 1
r3 = 1
r4 = 1

# Wagi ze strojenia
q1 =  1000
q2 =  399.9509149396778
q3 =  8463.704080454723
q4 =  3782.7224937855567
q5 =  7049.835414885605
q6 =  10000.0
q7 =  126.05153250830571
q8 =  1.0
q9 =  1.0
q10 =  6044.117005806847
q11 =  6339.704882247752
q12 =  9842.139809506909
r1 =  3.739653591961215
r2 =  1.0
r3 =  9.268321538645477
r4 =  1.0

q = np.array([q1, q2, q3, q4, q5, q6, q7, q8, q9 , q10, q11, q12])
r = np. array([r1, r2, r3, r4])

# Define system dynamics (example: simple inverted pendulum)
A = np.array([
    [0, 0, 0, 0, 0, 0, 1.0000, -0.0000, -0.0003, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.0000, 0.9971, -0.0762, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.0003, 0.0762, 0.9971, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, -0.0003],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9971, -0.0762],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0762, 0.9971],
    [0, 0, -0.0000, -0.0000, -32.1740, 0.0000, -0.0189, 0.0000, 0.0000, 0.0316, 0.0092, -0.0002],
    [0, 0, -0.0001, 32.0804, 0.0009, 0.0000, -0.0000, -0.0667, -0.0000, -0.0124, 0.0321, 0.1633],
    [0, 0, -0.0010, -2.4528, 0.0112, 0.0000, -0.0006, 0.0005, 0.3744, -0.0001, 0.0018, -0.3854],
    [0, 0, 0.0000, -0.0000, 0.0000, -0.0000, 0.6268, -0.0395, 0.0314, -9.2556, -59.4366, 0.0310],
    [0, 0, -0.0000, -0.0000, 0.0000, 0.0000, -0.0017, 0.4103, 0.0276, 38.9998, -3.7512, -0.0059],
    [0, 0, -0.0000, -0.0000, 0.0000, -0.0000, 0.0030, 0.1570, 0.0071, -0.0273, -0.2981, -0.6710]
])

B = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-0.0000, -0.0545, -0.0002, -0.0000],
    [0.0340, -0.0001,  0.0002,  0.2001],
    [0.0003,  0.0001, -0.5912, -0.0000],
    [0.0685,  2.3015,  0.0083,  0.0229],
    [-0.9379, -0.0165,  0.0061, -0.0222],
    [-0.0001,  0.0111, -0.1838, -0.6566]
])
Q = np.diag(q)

R = np.diag(r)

# Solve the continuous-time Algebraic Riccati Equation
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute LQR gain
K = np.linalg.inv(R) @ B.T @ P

# Simulate the system
def simulate_lqr(time, x0, A, B, K, X_des):
    x = np.zeros((len(time), len(x0)))
    u = np.zeros((len(time), 4))
    
    x[0] = x0
    dt = 0.01
    J = 0.0
    upp = 100*np.ones(4)
    u0 = np.array([50, 50, 50, 60])
    for i in range(1, len(time)):

        # LQR control law
        u[i] = -K @ (x[i-1] - X_des[i-1])  

        # Constraints
        u[i] = np.minimum(u[i], upp-u0)
        u[i] = np.maximum(u[i], -upp-u0)
        x[i] = x[i-1] + dt * (A @ x[i-1] + B @ u[i]).flatten()

        # Cost function
        J += (pow(x[i-1][0] - X_des[i-1][0],2) * dt) + (pow(x[i-1][1] - X_des[i-1][1],2) * dt) + (pow(x[i-1][2] - X_des[i-1][2],2) * dt)
    
    return x, u, J

# Initial state
x0 = np.zeros(12)
T = 10  # Total simulation time
dt = 0.01
time = np.arange(0, T, dt)

# Define trajectory
x_des = 30*np.ones(len(time))
y_des = 30*np.ones(len(time))
z_des = -10*np.ones(len(time))
phi_des = np.zeros(len(time))
theta_des = np.zeros(len(time))
psi_des = np.zeros(len(time))
vx_des = np.zeros(len(time))
vy_des = np.zeros(len(time))
vz_des = np.zeros(len(time))
p_des = np.zeros(len(time))
q_des = np.zeros(len(time))
r_des = np.zeros(len(time))
X_des = np.vstack((x_des, y_des, z_des, phi_des, theta_des, psi_des, vx_des, vy_des, vz_des, p_des, q_des, r_des)).T

# Run simulation
x, u, J = simulate_lqr(time, x0, A, B, K, X_des)

print("Całka z uchybu wynosi: ", J)

# Plot results for 12 state variables
fig, axs = plt.subplots(6, 2, figsize=(12, 18))

state_labels = [
    'x', 'y', 'z', 'phi', 'theta', 'psi', 
    'vx', 'vy', 'vz', 'p', 'q', 'r'
]

for i in range(12):
    ax = axs[i // 2, i % 2]
    if (state_labels[i]) == 'phi' or (state_labels[i]) == 'theta' or (state_labels[i]) == 'psi':
        ax.plot(time, 180/np.pi*x[:, i])
    else:
        ax.plot(time, x[:, i])
    ax.set_title(state_labels[i])
    #ax.set_xlabel('Time [s]')
    ax.set_ylabel('State Value')
    ax.grid(True)

plt.tight_layout()
plt.show()

# Plot results for control signals
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

control_labels = ['xa', 'xb', 'xc', 'xp']

for i in range(4):
    ax = axs[i // 2, i % 2]
    ax.plot(time, u[:, i])
    ax.set_title(control_labels[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Control Signal')
    ax.grid(True)

plt.tight_layout()
plt.show()
