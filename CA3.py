import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  
m = 1.0  
mu = 1
proposal_step = 0.01  
N = 100  
num_samples = 100  
thermalization = 10
beta = 1.0  
values = [
    (1.0,1.0,0),
    (2.0,2.0,0),
    (1.0,1.5,0),
    (1.0,1.0,1.0),
    (2.0,2.0,2.0),
    (1.0,1.5,3.0),
    (-1.0, 1.0, 1)
    
    ]


time_step = 1e-2  
time_values = np.linspace(0, N * time_step, N)
positions = np.zeros(N)


def V(x, mu, lambda_):
    return (0.5 * (mu**2) * x**2) + 3*(lambda_ * (x**4))


def compute_action(path, m, mu,lambda_, time_step):
    action = 0.0
    
    for i in range(1, len(path)):
        kinetic = 0.5 * m * ((path[i] - path[i-1]) / time_step)**2
        potential = V(path[i], mu, lambda_)
        action += (kinetic - potential) * time_step
    
    return action

for mu,m,lambda_ in values:
    def metropolis_path(N, proposal_step, num_samples, mu, m, lambda_, time_step):
        paths = []
        
        for _ in range(num_samples):
            path = np.zeros(N)
            for i in range(1, N):
                path[i] = path[i-1] + np.random.uniform(-proposal_step, proposal_step)
            
            current_action = compute_action(path, m, mu,lambda_, time_step)
            
            for step in range(N):
                new_path = path.copy()
                new_path[step] += np.random.uniform(-proposal_step, proposal_step)
                
                new_action = compute_action(new_path, m, mu,lambda_, time_step)
                delta_action = new_action - current_action
                
                if np.random.rand() < np.exp(-beta * delta_action):
                    path = new_path
                    current_action = new_action
            
            paths.append(path)
        
        return np.array(paths)
    
    paths = metropolis_path(N, proposal_step, num_samples, mu, m,lambda_, time_step)
    
    energies = []
    for path in paths:
        kinetic = 0.0
        potential = 0.0
        for i in range(1, len(path)):
            kinetic += 0.5 * m * ((path[i] - path[i-1]) / time_step)**2
            potential += V(path[i], mu, lambda_)
        
        total_energy = (kinetic + potential) / len(path)
        energies.append(total_energy)
    
    # Calculate the average ground state energy and first excited state energy
    ground_state_energy = np.mean(energies)
    first_excited_state_energy = 1.5 * ground_state_energy
    
    # Print the results
    print("for m = ", m, ", mu = ", mu, "lambda = ", lambda_)
    print(f"Estimated ground state energy: {ground_state_energy:.4f}")
    print(f"Estimated first excited state energy: {first_excited_state_energy:.4f}")
    
    # Plot the probability distribution of the ground state (histogram)
    # Flatten the positions from the samples
    flattened_positions = np.concatenate(paths)
    
    plt.hist(flattened_positions, bins=1000, density=True, alpha=0.7, label="Sampled Distribution")
    plt.xlabel('Position (x)')
    plt.ylabel('Probability Density')
    plt.title('Ground State Probability Distribution (Metropolis)')
    plt.grid(True)
    plt.legend()
    plt.show()

x_values = np.linspace(-1,1,100)
plt.plot(x_values, (V(x_values, -1, 1)))

 