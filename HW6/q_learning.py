#import os
#os.chdir("C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW6\\handout\\python")

from environment import MountainCar
import sys
import csv
import numpy as np


class Agent(object):
    def __init__(self, episodes, max_iterations, epsilon, gamma, learning_rate, car):
        self.episodes = episodes
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.car = car
        self.weights = np.zeros([self.car.state_space, self.car.action_space])#raw: 2*3, tile: 2048*3
        self.bias = 0.0
        self.action = None
        self.done = False
        self.reward_list = []

    #calculate Q value
    def update_Q(self, state):
        Q = []
        for i in range(len(self.weights[0])):
            tmp = 0.0
            for key in state:
                tmp += state[key] * self.weights[key][i]
            Q.append(tmp + self.bias)
        return Q
    
    # use the epsilon-greedy strategy for action selection
    def action_selection(self, Q):
        optimal_action = np.argmax(Q)
        test = np.random.uniform(0, 1)
        if test < self.epsilon:
            self.action = np.random.randint(3)
        else:
            self.action = optimal_action
            

    def update_gradient(self, state):
        gradient_w = np.zeros([self.car.state_space, self.car.action_space]) #2*3 or 2048*3
        for key in state:
            gradient_w[key][self.action] = state[key]
        return gradient_w
        
    def update_weights(self, TD_error, gradient_w, gradient_b):
        self.weights = self.weights - self.learning_rate * TD_error * gradient_w
        self.bias = self.bias - self.learning_rate * TD_error * gradient_b
        
    def train(self):
        for i in range(self.episodes):
            state = self.car.reset()
            total_reward = 0.0 
            for j in range(self.max_iterations):
                #print(state)
                curr_Q = self.update_Q(state)
                #print(curr_Q)
                self.action_selection(curr_Q)#update self.action
                #print(self.action)
                new_state, reward, done = self.car.step(self.action)
                new_Q = self.update_Q(new_state)#update Q value
                
                TD_target = reward + self.gamma * max(new_Q)
                
                TD_error = curr_Q[self.action] - TD_target
                
                gradient_w = self.update_gradient(state)
               # print(gradient_w)
                gradient_b = 1
                
                self.update_weights(TD_error, gradient_w, gradient_b)#update weights and bias
                
                #update Q value
                total_reward += reward
                self.done = done
                state = new_state
                if self.done:
                    break
                #print()
            self.reward_list.append(total_reward)


def write_out_return(file_name, reward):
    with open(str(file_name), "wt") as f:
        output = ""
        for item in reward:
            output += str(item) + "\n"
        f.write(output)
    print("written!")


def write_out_weights(file_name, w, b):
    with open(str(file_name), "wt") as f:
        output = ""
        output += str(b) + "\n"
        for i in range(len(w)):
            for j in range(len(w[0])):
                output += str(w[i][j]) + "\n"
        f.write(output)
    print("written!")
    
    
def main(args):
    
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])
    '''
    mode = "raw"
    episodes = 4
    max_iterations = 200
    epsilon = 0.05
    gamma = 0.99
    learning_rate = 0.01
    weight_out = "weight_out"
    returns_out = "return_out"
    '''
    my_car = MountainCar(mode)
    my_agent = Agent(episodes, max_iterations, epsilon, gamma, learning_rate, my_car)
    my_agent.train()
    
    reward_list = my_agent.reward_list
    bias = my_agent.bias
    weights = my_agent.weights
    print(reward_list)
    print(bias)
    print(weights)
    
    write_out_return(returns_out, reward_list)
    write_out_weights(weight_out, weights, bias)
    
    

if __name__ == "__main__":
    #argv: mode weight_out returns_out episodes max_iterations epsilon gamma learning_rate
    main(sys.argv)
    #main(1)
    
