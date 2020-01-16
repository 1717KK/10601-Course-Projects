
import sys
import numpy as np

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

def cal_init_prob(x, tag_dict):
    num_matrix = np.ones(len(tag_dict))
    first_words = x[:, 0]
    for w in first_words:
        num_matrix[int(w)] += 1

    pi_matrix = num_matrix/np.sum(num_matrix)

    return pi_matrix

def cal_trans_prob(y, tag_dict):
    trans_matrix = np.ones((len(tag_dict), len(tag_dict)))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]-1):
            sj = y[i][j]
            sk = y[i][j+1]
            if int(sj) != -1 and int(sk) != -1:
                trans_matrix[int(sj)][int(sk)] += 1

    for i in range(trans_matrix.shape[0]):
        trans_matrix[i, :] = trans_matrix[i, :]/np.sum(trans_matrix[i, :])

    return trans_matrix


def cal_emiss_prob(x, y, word_dict, tag_dict):
    emiss_matrix = np.ones((len(tag_dict), len(word_dict)))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            word = x[i][j]
            state = y[i][j]
            if int(word) != -1 and int(state) != -1:
                emiss_matrix[int(state)][int(word)] += 1

    for i in range(emiss_matrix.shape[0]):
        emiss_matrix[i, :] = emiss_matrix[i, :] / np.sum(emiss_matrix[i, :])

    return emiss_matrix


def write_out(file_name, matrix):
    np.savetxt(file_name, matrix)
    print("written!")

if __name__ == "__main__":
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    word_dict = process_index(index_to_word)
    tag_dict = process_index(index_to_tag)
    x, y = process_data(train_input, word_dict, tag_dict)

    init_prob = cal_init_prob(y, tag_dict)
    trans_prob = cal_trans_prob(y, tag_dict)
    emiss_prob = cal_emiss_prob(x, y, word_dict, tag_dict)


    write_out(hmmprior, init_prob)
    write_out(hmmtrans, trans_prob)
    write_out(hmmemit, emiss_prob)





