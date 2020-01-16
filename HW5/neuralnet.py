import csv
import numpy as np
import os
import sys
import math

### read file

def read(filename):
    with open(str(filename), "rt") as f:
        input_file = f.readlines()

        y = []
        x = np.ones((len(input_file), 129)) #x0 = 1

        for i in range(len(input_file)):
            label = input_file[i][0]
            y.append(int(label))
            pixel = input_file[i][2:]
            pixel = pixel.strip().split(",")
            pixel = [int(item) for item in pixel]
            x[i, 1:] = pixel
            
    return x, y
    
### initialization

def init(flag, D, M):
    #D: hidden units
    #M: number of features = 128+1
    if flag == 1:
        #The weights are initialized randomly from a uniform distribution from -0.1 to 0.1
        #The bias parameters are initialized to zero
        alpha = np.random.uniform(-0.1, 0.1, (D, M))
        alpha[:, 0] = 0.0
        beta = np.random.uniform(-0.1, 0.1, (10, D + 1))
        beta[:, 0] = 0.0
    elif flag == 2:
        #All weights are initialized to 0
        alpha = np.zeros((D, M))
        beta = np.zeros((10, D + 1))
    return alpha, beta
    
### train

def linear_forward(a, weight):
    b = np.dot(weight, a)
    return b
    
def sigmoid_forward(a):
    b = 1 / (1 + np.exp(-a))
    return b
    
def softmax_forward(a):
    tmp = np.sum(np.exp(a))
    b = np.divide(np.exp(a), tmp)
    return b
    
def cross_entropy_forward(a, a_hat):
    b = -np.dot(a.T, np.log(a_hat))
    return b
    
def linear_backward(a, w, g_b):
    a.resize(len(a), 1)
    g_w = np.dot(g_b.T, a.T)
    g_a = np.dot(w.T, g_b.T)
    return g_w, g_a
    
def sigmoid_backward(b, g_b):
    tmp = np.multiply(b, 1-b)
    result = np.multiply(g_b, tmp)
    return result[1:]
    
def softmax_backward(b, g_b):
    g_b.resize(len(g_b), 1)
    b_reshape = np.reshape(b, (b.shape[0],1))
    tmp1 = np.dot(b_reshape, b_reshape.T)
    tmp2 = np.subtract(np.diag(b), tmp1)
    result = np.dot(g_b.T, tmp2)
    return result

def cross_entropy_backward(a, a_hat, g_b):
    tmp = np.divide(a, a_hat)
    g_a_hat = -tmp #g_b = 1
    return g_a_hat


def NN_Forward(x, y, alpha, beta):
    a = linear_forward(x, alpha)
    z = sigmoid_forward(a)
    z = np.append(1, z) #z0 = 1
    b = linear_forward(z, beta) 
    y_hat = softmax_forward(b) # row vector
    J = cross_entropy_forward(y, y_hat)
    
    return x, a, b, z, y_hat, J
    
def NN_Backward(x, y, alpha, beta, a, b, z, y_hat, J):
    g_J = 1
    g_y_hat = cross_entropy_backward(y, y_hat, g_J) #g_J = 1
    g_b = softmax_backward(y_hat, g_y_hat) # row vector
    g_beta, g_z = linear_backward(z, beta, g_b)
    g_a = sigmoid_backward(z, g_z)
    g_a.resize(1, len(g_a))
    g_alpha, g_x = linear_backward(x, alpha, g_a)
    return g_alpha, g_beta
    
def cross_entropy(x_set, y_set, alpha, beta):
    entropy = 0.0
    total = x_set.shape[0]
    for i in range(total):
        x = x_set[i, :] # ith row
        label = y_set[i]
        y = np.zeros(10)
        y[label] = 1
        J = NN_Forward(x, y, alpha, beta)[-1]
        entropy += J
    mean = entropy / total
    return mean
        
        
def NN_SGD(x_train, y_train, init_flag, hidden_units, num_epochs, learning_rate):
    alpha, beta = init(init_flag, hidden_units, x_train.shape[1])
    train_mean_ce_list = []
    test_mean_ce_list = []
    
    for i in range(num_epochs):
        for j in range(x_train.shape[0]):
            x = x_train[j, :] # jth row
            label = y_train[j]
            y = np.zeros(10)
            y[label] = 1
            x, a, b, z, y_hat, J = NN_Forward(x, y, alpha, beta)
            g_alpha, g_beta = NN_Backward(x, y, alpha, beta, a, b, z, y_hat, J)
            alpha = alpha - learning_rate * g_alpha
            beta = beta - learning_rate * g_beta
        
        train_mean_ce = cross_entropy(x_train, y_train, alpha, beta)
        test_mean_ce = cross_entropy(x_test, y_test, alpha, beta)
        train_mean_ce_list.append(train_mean_ce)
        test_mean_ce_list.append(test_mean_ce)
            
    return alpha, beta, train_mean_ce_list, test_mean_ce_list

### predict
def predict(x_set, y_set, alpha, beta):
    label_predict = []
    for i in range(x_set.shape[0]):
        x = x_set[i, :] # ith row
        label = y_set[i]
        y = np.zeros(10)
        y[label] = 1
        y_hat = NN_Forward(x, y, alpha, beta)[-2]
        l = np.argmax(y_hat)
        label_predict.append(l)
    return label_predict

def error(y_pred, y_real):
    total = len(y_real)
    count = 0
    for i in range(total):
        if y_pred[i] != y_real[i]:
            count += 1
    rate = count/total
    return rate

### write out

def write_out_label(label, file_name):
    with open(str(file_name), "wt") as f:
        output = ""
        for i in range(len(label)):
            output += str(label[i])
            output += "\n"
        f.write(output)
    print("written!")

def write_out_metric(train_ce, test_ce, error_train, error_test, file_name):
    output = ""
    epoch = len(train_ce)
    for i in range(epoch):
        output += "epoch=" + str(i+1) + " crossentropy(train): " + str(train_ce[i]) + "\n"
        output += "epoch=" + str(i+1) + " crossentropy(test): " + str(test_ce[i]) + "\n"
    output += "error(train): " + str(error_train) + "\n"
    output += "error(test): " + str(error_test) + "\n"
    with open(str(file_name), "wt") as f:
        f.write(output)
    print("written!")
    
###

if __name__ == "__main__":
    
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epochs = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    #prepare data
    x_train, y_train = read(train_input)
    x_test, y_test = read(test_input)
    
    #train the NN model
    alpha, beta, train_mean_ce, test_mean_ce = NN_SGD(x_train, y_train, init_flag, hidden_units, num_epochs, learning_rate)
    
    #predict
    y_pred_train = predict(x_train, y_train, alpha, beta)
    y_pred_test = predict(x_test, y_test, alpha, beta)
    
    #calulate error
    error_train = error(y_pred_train, y_train)
    error_test = error(y_pred_test, y_test)
    
    #write out label
    write_out_label(y_pred_train, train_out)
    write_out_label(y_pred_test, test_out)
    
    print(train_mean_ce)
    print(test_mean_ce)
    print(error_train)
    print(error_test)
    #write out metrics
    write_out_metric(train_mean_ce, test_mean_ce, error_train, error_test, metrics_out)
    
    