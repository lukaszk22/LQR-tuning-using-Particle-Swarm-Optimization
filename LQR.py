import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Define system dynamics (example: simple inverted pendulum)
A = np.array([[0, 1],
              [0, -0.5]])  # State matrix
B = np.array([[0],
              [1]])  # Input matrix
Q = np.array([[119.98222779820782, 0],
              [0, 1]])  # State cost matrix
R = np.array([[13.668304723145768]])  # Control cost matrix

# Solve the continuous-time Algebraic Riccati Equation
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute LQR gain
K = np.linalg.inv(R) @ B.T @ P

# Simulate the system
def simulate_lqr(time, x0, A, B, K, X_des):
    x = np.zeros((len(time), len(x0)))
    u = np.zeros((len(time), 1))
    
    x[0] = x0
    dt = 0.01
    J = 0.0
    for i in range(1, len(time)):

        # LQR control law
        u[i] = -K @ (x[i-1] - X_des[i-1])  
        # Ograniczenia
        u[i] = min(u[i], 5)
        u[i] = max(u[i], -5)

        x[i] = x[i-1] + dt * (A @ x[i-1] + B @ u[i]).flatten()

        J += sum(pow(x[i-1] - X_des[i-1],2) * dt)
    
    return x, u, J

# Initial state
x0 = np.array([0, 0])  # Initial position and velocity
T = 10  # Total simulation time
dt = 0.01
time = np.arange(0, T, dt)
x_des = np.ones(len(time))
v_des = np.zeros(len(time))
X_des = np.vstack((x_des, v_des)).T

# Run simulation
x, u, J = simulate_lqr(time, x0, A, B, K, X_des)

print("Całka z uchybu wynosi: ", J)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time, x[:, 0], label='Position')
plt.plot(time, x_des)
plt.xlabel('Time [s]')
plt.ylabel('State')
plt.legend()
plt.title('LQR Controlled System')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time, u, label='Control')
plt.xlabel('Time [s]')
plt.ylabel('Control')
plt.legend()
plt.title('LQR Controlled System')
plt.grid()
plt.show()
