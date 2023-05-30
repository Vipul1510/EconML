import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)

N = 4
eta = 0.5
gamma = 2

# Values of horizons
Values_of_T = np.arange(100, 1001, 10)
# To store regret associated with each horizon
Values_of_regret = np.zeros(len(Values_of_T))

# It returns reward of and regret associated with round t and arm_index'index' 
def get_reward(arm_index, Regret):
    points = np.random.uniform(-1.0, 1.0, size=N)
    Regret= Regret + np.max(points) - points[arm_index]
    return points[arm_index], Regret


for j in range(len(Values_of_T)):

    # Initialization of required quantities
    T = Values_of_T[j]
    weights = np.full((N,),1.0)
    R = np.zeros(N,)
    Regret = 0

    for i in range(T):

        # Finding prob. by normalizing weights.
        P = weights/sum(weights)  

        # Finding arm index using given probabilities
        indices = np.arange(len(P))                   
        selected_arm = np.random.choice(indices, p = P)    

        # Getting reward
        reward , Regret = get_reward(selected_arm, Regret)

        # Finding ~L 
        R = np.zeros(N,)
        R[selected_arm] = reward/(P[selected_arm] + gamma)

        # Updating weights
        weights = weights * np.exp(eta*R)
    
    # Saving Value of regret 
    Values_of_regret[j] = Regret

# Plotting regret vs number of rounds plot
plt.plot(Values_of_T,Values_of_regret)
plt.xlabel('Time horizon')
plt.ylabel('Regret')
plt.title('Regret vs Time horizon')
plt.show()