import numpy as np
import os
import sys

### learning the model
    
# generate the list of label and list of words
def readFile(fileName):
    with open(str(fileName), "rt") as f:
        file_input = f.readlines()
        label_list = []
        words_list = []
        for i in range(len(file_input)):
            label = file_input[i][0]
            attribute = file_input[i][2:]
            words = attribute.strip().split("\t")
            label_list.append(int(label))
            word_dict = dict()
            for word in words:
                if word[:-2] not in word_dict:
                    word_dict[word[:-2]] = int(word[-1])
            words_list.append(word_dict)
    return label_list, words_list
    
# generate initial w (vector of zero) and b=0
def generate_init_w_b(fileName):
    with open(str(fileName), 'r') as f:
        dict_list = f.readlines()
    w = np.zeros((len(dict_list), 1))
    b = 0
    return w, b
    
def cal_Sigmoid(u):
    result = 1 / (1 + np.exp(-u))
    return result
    
# calculate sparse dot for ith row training example
def sparse_dot(X, W, b):
    product = 0.0
    for key in X:
        product += X[key] * W[int(key)]
    product += b
    return product

# calculate SGD for ith row training example
def cal_SGD(x_i, y, dot_product, w):
    w_gradient = np.zeros((len(w), 1))
    sigmoid = cal_Sigmoid(dot_product)[0]
    b_gradient  = (sigmoid - y ) * 1
    for k, v in x_i.items():
        w_gradient[int(k)] = (sigmoid - y) * v
    return w_gradient, b_gradient 

# use lr model to learn weight and bias
def lr_train(train_label_list, train_words_list, init_w, init_b, num_epoch, learning_rate):
    for j in range(num_epoch):
        for i in range(len(train_words_list)): # row = i
            dot_product = sparse_dot(train_words_list[i], init_w, init_b)# for ith row training example
            w_gradient, b_gradient = cal_SGD(train_words_list[i], train_label_list[i], dot_product, init_w)
            delta_w = learning_rate * w_gradient
            delta_b = learning_rate* b_gradient
            init_w = np.subtract(init_w, delta_w)
            init_b = init_b - delta_b
    return init_w, init_b

### prediction and write out

# use learned w and b to predict labels
def predict_label(y, words, w, b):
    labels = []
    for word in words:
        dot_product = sparse_dot(word, w, b)
        label = np.round(cal_Sigmoid(dot_product))[0]
        labels.append(int(label))
    return labels


# use predicted label and real label to calculate its error
def cal_error(real_label, predicted_label):
    count = 0
    total = len(real_label)
    for i in range(total):
        if real_label[i] != predicted_label[i]:
            count += 1
    error = float(count/total)
    return error

def write_out_label(label, file_name):
    with open(str(file_name), "wt") as f:
        output = ""
        for i in range(len(label)):
            output += str(label[i])
            output += "\n"
        f.write(output) 
    print("written!")
    
def write_out_metric(train, test, file_name):
    with open(str(file_name), "wt") as f:
        output = ""
        output += "error(train): " + str(train) + "\n"
        output += "error(test): " + str(test) + "\n"
        f.write(output)
    print("written!")


### read the arguments and do operations

if __name__ == "__main__":

    formatted_train_out = sys.argv[1]
    formatted_validation_out = sys.argv[2]
    formatted_test_out = sys.argv[3]
    dict_input  = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])

    # prepare label list and words dictionary
    real_train_labels, real_train_words = readFile(formatted_train_out)
    
    real_test_labels, real_test_words = readFile(formatted_test_out)
    
    # generate initial weights and bias
    init_w, init_b = generate_init_w_b(dict_input)
    
    # use lr model to learn weights and bias
    learned_W, learned_b = lr_train(real_train_labels, real_train_words, init_w, init_b, num_epoch, 0.1)
    
    # use learned weights and bias to predict labels
    predicted_train_labels = predict_label(real_train_labels, real_train_words, learned_W, learned_b)
    #print(predicted_train_labels)
    
    predicted_test_labels = predict_label(real_test_labels, real_test_words, learned_W, learned_b)
    
    # calculate error between real labels and predicted labels
    train_error = cal_error(real_train_labels, predicted_train_labels)
    #print(train_error)
    test_error = cal_error(real_test_labels, predicted_test_labels)
    #print(test_error)
    
    #write out predicted labels
    write_out_label(predicted_train_labels, str(train_out))
    write_out_label(predicted_test_labels, str(test_out))
    
    #write out error
    write_out_metric(train_error, test_error, str(metrics_out))


