import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.linalg

DIMENSIONS = 2              # Number of dimensions
GLOBAL_BEST = 0.0           # Global Best of Cost function
B_LO = 1                    # Upper boundary of search space
B_HI = 1000                 # Upper boundary of search space

POPULATION = 20             # Number of particles in the swarm
V_MAX = 100                 # Maximum velocity value
PERSONAL_C = 1.0            # Personal coefficient factor
SOCIAL_C =  2.0             # Social coefficient factor
CONVERGENCE = 0.001         # Convergence value
MAX_ITER = 100              # Maximum number of iterrations

# Particle class
class Particle():
    def __init__(self, x, y, z, velocity):
        self.pos = [x, y]
        self.pos_z = z
        self.velocity = velocity
        self.best_pos = self.pos.copy()

class Swarm():
    def __init__(self, pop, v_max):
        self.particles = []             # List of particles in the swarm
        self.best_pos = None            # Best particle of the swarm
        self.best_pos_z = math.inf      # Best particle of the swarm

        for _ in range(pop):
            x = np.random.uniform(B_LO, B_HI)
            y = np.random.uniform(B_LO, B_HI)
            z = cost_function(x, y)
            velocity = np.random.rand(2) * v_max
            particle = Particle(x, y, z, velocity)
            self.particles.append(particle)
            if self.best_pos != None and particle.pos_z < self.best_pos_z:
                self.best_pos = particle.pos.copy()
                self.best_pos_z = particle.pos_z
            else:
                self.best_pos = particle.pos.copy()
                self.best_pos_z = particle.pos_z

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


def cost_function(q1, r1):

    r1 = max(r1, 1)
    q1 = max(q1, 1)

    # Define system dynamics (example: simple inverted pendulum)
    A = np.array([[0, 1],
                [0, -0.5]])  # State matrix
    
    B = np.array([[0],
                [1]])  # Input matrix
    
    Q = np.array([[q1, 0],
                [0, 1]])  # State cost matrix
    
    R = np.array([[r1]])  # Control cost matrix

    # Solve the continuous-time Algebraic Riccati Equation
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # Compute LQR gain
    K = np.linalg.inv(R) @ B.T @ P

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

    return J

def particle_swarm_optimization():

    # Initialize plotting variables
    x = np.linspace(B_LO, B_HI, 50)
    y = np.linspace(B_LO, B_HI, 50)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure("Particle Swarm Optimization")

    # Initialize swarm
    swarm = Swarm(POPULATION, V_MAX)

    # Initialize inertia weight
    inertia_weight = 0.5 + (np.random.rand()/2)
    
    curr_iter = 0
    iters = []
    scores = []
    while curr_iter < MAX_ITER:

        for particle in swarm.particles:

            for i in range(0, DIMENSIONS):
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                
                # Update particle's velocity
                personal_coefficient = PERSONAL_C * r1 * (particle.best_pos[i] - particle.pos[i])
                social_coefficient = SOCIAL_C * r2 * (swarm.best_pos[i] - particle.pos[i])
                new_velocity = inertia_weight * particle.velocity[i] + personal_coefficient + social_coefficient

                # Check if velocity is exceeded
                if new_velocity > V_MAX:
                    particle.velocity[i] = V_MAX
                elif new_velocity < -V_MAX:
                    particle.velocity[i] = -V_MAX
                else:
                    particle.velocity[i] = new_velocity

            # Update particle's current position
            particle.pos += particle.velocity
            particle.pos_z = cost_function(particle.pos[0], particle.pos[1])

            # Update particle's best known position
            if particle.pos_z < cost_function(particle.best_pos[0], particle.best_pos[1]):
                particle.best_pos = particle.pos.copy()

                # Update swarm's best known position
                if particle.pos_z < swarm.best_pos_z:
                    swarm.best_pos = particle.pos.copy()
                    swarm.best_pos_z = particle.pos_z
                    
            # Check if particle is within boundaries
            if particle.pos[0] > B_HI:
                particle.pos[0] = np.random.uniform(B_LO, B_HI)
                particle.pos_z = cost_function(particle.pos[0], particle.pos[1])
            if particle.pos[1] > B_HI:
                particle.pos[1] = np.random.uniform(B_LO, B_HI)
                particle.pos_z = cost_function(particle.pos[0], particle.pos[1])
            if particle.pos[0] < B_LO:
                particle.pos[0] = np.random.uniform(B_LO, B_HI)
                particle.pos_z = cost_function(particle.pos[0], particle.pos[1])
            if particle.pos[1] < B_LO:
                particle.pos[1] = np.random.uniform(B_LO, B_HI)
                particle.pos_z = cost_function(particle.pos[0], particle.pos[1])

        # Check for convergence
        if abs(swarm.best_pos_z - GLOBAL_BEST) < CONVERGENCE:
            print("The swarm has met convergence criteria after " + str(curr_iter) + " iterrations. Best is "+str(swarm.best_pos_z))
            break
        curr_iter += 1
        iters.append(curr_iter)
        scores.append(swarm.best_pos_z)
    print("END iterations. Best is "+str(swarm.best_pos_z))
    #plt.show()
    return iters, scores, swarm.best_pos

if __name__ == "__main__":

    start_time = time.time()

    iters, scores, best_pos = particle_swarm_optimization()

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print("q1 = ", best_pos[0])
    print("r1 = ", best_pos[1])

    plt.figure(figsize=(10, 6))
    plt.plot(iters, scores, linestyle='-', color='b', label='f(x)')
    plt.xlabel('Numer iteracji', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Wartość funkcji kosztu w funkcji iteracji (PSO)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()