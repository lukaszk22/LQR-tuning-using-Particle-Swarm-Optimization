import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.linalg

DIMENSIONS = 15             # Number of dimensions
GLOBAL_BEST = 0.0           # Global Best of Cost function
B_LO = 1                    # Upper boundary of search space
B_HI = 10000                # Upper boundary of search space

POPULATION = 20             # Number of particles in the swarm
V_MAX = 20                  # Maximum velocity value
PERSONAL_C = 1.0            # Personal coefficient factor
SOCIAL_C =  2.0             # Social coefficient factor
CONVERGENCE = 0.001         # Convergence value
MAX_ITER = 1000              # Maximum number of iterrations

# Particle class
class Particle():
    def __init__(self):
        self.pos = np.random.uniform(B_LO, B_HI, DIMENSIONS)
        self.velocity = np.random.uniform(-V_MAX, V_MAX, DIMENSIONS)
        self.best_pos = self.pos.copy()
        self.pos_cost = cost_function(*self.pos)
        self.best_pos_cost = self.pos_cost

class Swarm():
    def __init__(self, pop):
        self.particles = [Particle() for _ in range(pop)]
        self.best_pos = self.particles[0].best_pos.copy()
        self.best_pos_cost = self.particles[0].best_pos_cost
        for p in self.particles:
            if p.best_pos_cost < self.best_pos_cost:
                self.best_pos = p.best_pos.copy()
                self.best_pos_cost = p.best_pos_cost

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
        u[i] = np.maximum(u[i], -u0)

        # Simulation step
        x[i] = x[i-1] + dt * (A @ x[i-1] + B @ u[i]).flatten()

        # Cost function update
        J += (pow(x[i-1][0] - X_des[i-1][0],2) * dt) + (pow(x[i-1][1] - X_des[i-1][1],2) * dt) + (pow(x[i-1][2] - X_des[i-1][2],2) * dt) + 180/np.pi*(pow(x[i-1][3] - X_des[i-1][3],2) * dt) + 180/np.pi*(pow(x[i-1][4] - X_des[i-1][4],2) * dt) + 180/np.pi*(pow(x[i-1][5] - X_des[i-1][5],2) * dt)
    
    return x, u, J


def cost_function(*params):
    q1 = 1000
    q = np.array([q1] + list(params[:11]))
    r = np.array(params[11:])

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

    return J

def particle_swarm_optimization():
    swarm = Swarm(POPULATION)
    inertia_weight = 0.5 + (np.random.rand() / 2)
    iters, scores = [], []

    for curr_iter in range(MAX_ITER):
        for particle in swarm.particles:
            r1, r2 = np.random.rand(DIMENSIONS), np.random.rand(DIMENSIONS)
            personal_component = PERSONAL_C * r1 * (particle.best_pos - particle.pos)
            social_component = SOCIAL_C * r2 * (swarm.best_pos - particle.pos)
            particle.velocity = inertia_weight * particle.velocity + personal_component + social_component
            particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)
            particle.pos += particle.velocity
            particle.pos = np.clip(particle.pos, B_LO, B_HI)
            particle.pos_cost = cost_function(*particle.pos)
            
            if particle.pos_cost < particle.best_pos_cost:
                particle.best_pos = particle.pos.copy()
                particle.best_pos_cost = particle.pos_cost
                if particle.best_pos_cost < swarm.best_pos_cost:
                    swarm.best_pos = particle.best_pos.copy()
                    swarm.best_pos_cost = particle.best_pos_cost

        if abs(swarm.best_pos_cost - GLOBAL_BEST) < CONVERGENCE:
            break
        iters.append(curr_iter)
        scores.append(swarm.best_pos_cost)
    return iters, scores, swarm.best_pos

if __name__ == "__main__":

    start_time = time.time()

    iters, scores, best_pos = particle_swarm_optimization()

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Cost function value (tuning): {iters[-1]:.6f}")
    print("q1 = ", 1000)
    print("q2 = ", best_pos[0])
    print("q3 = ", best_pos[1])
    print("q4 = ", best_pos[2])
    print("q5 = ", best_pos[3])
    print("q6 = ", best_pos[4])
    print("q7 = ", best_pos[5])
    print("q8 = ", best_pos[6])
    print("q9 = ", best_pos[7])
    print("q10 = ", best_pos[8])
    print("q11 = ", best_pos[9])
    print("q12 = ", best_pos[10])
    print("r1 = ", best_pos[11])
    print("r2 = ", best_pos[12])
    print("r3 = ", best_pos[13])
    print("r4 = ", best_pos[14])

    plt.figure(figsize=(10, 6))
    plt.plot(iters, scores, linestyle='-', color='b', label='f(x)')
    plt.xlabel('Numer iteracji', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Wartość funkcji kosztu w funkcji iteracji (PSO)', fontsize=14)
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

    ##################################################################################################

    q = np.array([1000] + list(best_pos[:11]))
    r = np.array(best_pos[11:])

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
            u[i] = np.maximum(u[i], -u0)
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

    print("Całka z uchybu wynosi (walidacja): ", J)

    # Plot results for 12 state variables
    fig, axs = plt.subplots(6, 2, figsize=(12, 18))

    state_labels = [
        'x', 'y', 'z', 'phi', 'theta', 'psi', 
        'vx', 'vy', 'vz', 'p', 'q', 'r'
    ]

    for i in range(12):
        ax = axs[i // 2, i % 2]
        if i >2 & i < 6:
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