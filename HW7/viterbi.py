import sys
import time

import numpy as np
import math

def process_index(file_name):
    with open(str(file_name)) as f:
        d = dict()
        idx = 0
        file = f.readlines()
        for word in file:
            word = word.strip()
            if word not in d:
                d[word] = idx
            idx += 1
    return d

def process_data(file_name, word_dict, tag_dict):
    with open(str(file_name)) as f:
        file = f.readlines()
        row = len(file)
        max_col = 0
        for i in range(len(file)):
            r = file[i].strip()
            pairs = r.split(" ")
            col = len(pairs)
            if col > max_col:
                max_col = col
        x = -np.ones((row, max_col))
        y = -np.ones((row, max_col))

        for i in range(len(file)):
            row = file[i].strip()
            pairs = row.split(" ")
            for j in range(len(pairs)):
                x_y = pairs[j].split("_")
                x[i][j] = word_dict[x_y[0]]
                y[i][j] = tag_dict[x_y[1]]
    return x, y


def viterbi(sequence, prior, emit, trans):
    #prior's length:9
    #trans's shape:(9, 9)
    #emit's shape: (9, 8127)
    T = len(sequence)
    N = len(prior)
    viterbi = np.zeros((N, T))
    back_pointers = np.zeros((N, T))
    #initialization step
    for s in range(N):
        viterbi[s][0] = math.log(prior[s]) +math.log(emit[s][int(sequence[0])])

    #recursion step
    for t in range(1, T):
        for s in range(N):
            tmp_vector = np.zeros(N)
            for pre_s in range(N):
                lw = viterbi[pre_s][t-1]
                a = math.log(trans[pre_s][s])
                b = math.log(emit[s][int(sequence[t])])
                tmp_vector[pre_s] = lw+a+b
            viterbi[s][t] = np.max(tmp_vector)
            back_pointers[s][t] = np.argmax(tmp_vector)

    # termination step
    best_path_pointer = []
    y_hat = int(np.argmax(viterbi[:, -1]))
    best_path_pointer.append(y_hat)
    for t in range(T-1, 0, -1):
        y_hat = back_pointers[int(y_hat)][t]
        best_path_pointer.append(int(y_hat))

    best_path_pointer.reverse()

    return best_path_pointer

def predict(words, pointer, word_dict, tag_dict):
    pred_lst = []
    word_lst = []
    tag_lst = []
    for key in word_dict:
        word_lst.append(key)
    for key in tag_dict:
        tag_lst.append(key)

    for i in range(len(words)):
        word = word_lst[int(words[i])]
        idx = pointer[i]
        tag = tag_lst[int(idx)]
        pred = str(word) + "_" + str(tag)
        pred_lst.append(pred)
    return pred_lst

def write_out_predict(file_name, lst):
    with open(str(file_name), "wt") as f:
        result = ""
        for i in range(len(lst)):
            line = ""
            for j in range(len(lst[i])):
                line += lst[i][j] + " "
            line = line.strip()
            line += "\n"
            result += line
        f.write(result)
    print("written!")

def cal_accuracy(pred, test_file):
    with open(str(test_file)) as f:
        file = f.readlines()
        count = 0
        total = 0
        for i in range(len(file)):
            line = file[i].strip()
            line = line.split(" ")
            for j in range(len(line)):
                x_y = line[j].split("_")
                real_tag = x_y[1]
                pred_x_y = pred[i][j].split("_")
                pred_tag = pred_x_y[1]
                if real_tag == pred_tag:
                    count += 1
            total += len(pred[i])
        accuracy = count/total
    return accuracy

def write_out_metric(file_name, accuracy):
    with open(str(file_name), "wt") as f:
        result = "Accuracy: " + str(accuracy)
        f.write(result)
    print("written!")


if __name__ == "__main__":
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted = sys.argv[7]
    metric_file = sys.argv[8]

    word_dict = process_index(index_to_word)
    tag_dict = process_index(index_to_tag)
    x, y = process_data(test_input, word_dict, tag_dict)
    prior_matrix = np.loadtxt(hmmprior) # shape:(9,)
    trans_matrix = np.loadtxt(hmmtrans) # shape:(9, 9)
    emiss_matrix = np.loadtxt(hmmemit) # shape: (9, 8127)

    predict_lst = []
    for i in range(x.shape[0]):
        line = x[i,:]
        count = np.sum(line==-1)
        T = len(line) - count
        words = line[0:T]
        best_path_pointer = viterbi(words, prior_matrix, emiss_matrix, trans_matrix)
        predict_line = predict(words, best_path_pointer, word_dict, tag_dict)
        predict_lst.append(predict_line)

    accuracy = cal_accuracy(predict_lst, test_input)

    write_out_predict(predicted, predict_lst)
    write_out_metric(metric_file, accuracy)




