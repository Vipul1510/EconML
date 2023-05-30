import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)


class LinUCB:
    def __init__(self, num_arms, num_features, alpha):
        self.num_arms = num_arms
        self.num_features = num_features
        self.alpha = alpha
        self.A = [np.identity(num_features) for _ in range(num_arms)] #confidence matrix for arms
        self.b = [np.zeros((num_features, 1)) for _ in range(num_arms)] #reward vector for arms
        self.theta = [np.zeros((num_features, 1)) for _ in range(num_arms)] #parameter vector arm: Represents estimated relationship between features and expected reward
        self.regret = []

    def select_arm(self, context):
        p = [np.dot(self.theta[a].T, context) + self.alpha * np.sqrt(np.dot(context.T, np.dot(np.linalg.inv(self.A[a]), context))) for a in range(self.num_arms)] #calculating UCB for each arm
        return np.argmax(p)
    
    def update(self, arm, context, reward):
        self.A[arm] += np.dot(context, context.T)
        self.b[arm] += reward * context
        self.theta[arm] = np.dot(np.linalg.inv(self.A[arm]), self.b[arm])
        
    def calculate_regret(self, reward, optimal_reward):
        self.regret.append(optimal_reward - reward)
        return np.sum(self.regret)
    
    def reset(self):
        self.A = [np.identity(num_features) for _ in range(num_arms)]
        self.b = [np.zeros((num_features, 1)) for _ in range(num_arms)]
        self.theta = [np.zeros((num_features, 1)) for _ in range(num_arms)]
        self.regret = []


# Hardcoded values
num_arms = 5
num_features = 10
alpha = 0.1

# Instance of LinUCB
linucb = LinUCB(num_arms, num_features, alpha)

# Simulating the bandit environment
optimal_reward = 0
mu_true = np.array([0.5, 0.2]) 
D = np.array([[0, 1], [0.25, 0.75], [0, 0.25]]) 

# Values of horizons
Values_of_T = np.arange(100, 1001, 10)
# To store regret associated with each horizon
Values_of_regret = np.zeros(len(Values_of_T))


for j in range(len(Values_of_T)):

    num_rounds=Values_of_T[j]
    # For specified time horizon
    for t in range(1, num_rounds + 1):

        # Selecting the arm
        context = np.random.rand(num_features, 1)  # Generate random context
        arm = linucb.select_arm(context)
        xt_idx = np.random.randint(len(D))  # Randomly choose an index

        # Finding reward for the selected arm
        xt = D[xt_idx] 
        reward = np.dot(mu_true, xt) + np.random.uniform(-1, 1); 

        # Updating the values 
        linucb.update(arm, context, reward)

        # Updating the optimal reward
        optimal_reward = optimal_reward + np.max([np.dot(linucb.theta[a].T, context) for a in range(num_arms)])
    
    # Finding cummulative reward
    Values_of_regret[j]=linucb.calculate_regret(reward, optimal_reward)

    # Resetting the linucb instance for next time horizon
    linucb.reset()
    optimal_reward=0

# Plotting regret vs number of rounds plot
plt.plot(Values_of_T,Values_of_regret)
plt.xlabel('Time horizon')
plt.ylabel('Regret')
plt.title('Regret vs Time horizon')
plt.show()

