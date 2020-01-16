import csv
import sys
import os

os.chdir("C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW4")

dict_address = "C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW4\\handout\\dict.txt"
# train_data_address = "C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW4\\handout\\smalldata\\smalltrain_data.tsv"
# valid_data_address = "C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW4\\handout\\smalldata\\smallvalid_data.tsv"
# test_data_address = "C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW4\\handout\\smalldata\\smalltest_data.tsv"
large_train = "C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW4\\handout\\largedata\\train_data.tsv"
large_valid = "C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW4\\handout\\largedata\\valid_data.tsv"
large_test = "C:\\Yiqi\\CMU_in_the_work\\10601\\F19_10601_HW4\\handout\\largedata\\test_data.tsv"


###deal with the data

def prepareDict(address):
    
    with open(str(address), "rt") as f1:
        dict_file = csv.reader(f1)
        dict_list = [row1 for row1 in dict_file]
        dictionary = dict()
        for i in range(len(dict_list)):
            new_list = dict_list[i][0].split()
            if new_list[0] not in dictionary:
                dictionary[new_list[0]] = new_list[1] 
                
    return dictionary
            
#def parseData(dict_input, train_address,formatted_train_out, t):
def parseData(dict_input, address, output_file, t):
    
    file_input = dict()
    with open(str(address), 'r') as f1:
        file_input = f1.readlines()
        #print(file_input[0])
        word_list = []
        label_list = []
        for i in range(len(file_input)):
            label, attribute = file_input[i].split("\t")
            words = attribute.strip().split(" ")
            word_dict = dict()
            for word in words:
                try:
                    index = dict_input[word]
                    #print(word)
                    #print(index)
                except:
                    continue
                
                if t == 1:
                    if index not in word_dict:
                        #print(index)
                        #print(word)
                        word_dict[index] = 1
                        
                elif t == 4:
                    
                    if index not in word_dict:
                        word_dict[index] = 1
                    else:
                        word_dict[index] += 1
                        
            
            if t == 4:           
                new_word_dict = dict()
                for word in word_dict:
                    if word_dict[word] >= 4:
                        continue
                    else:
                        new_word_dict[word] = 1
            
            if t == 1:
                word_list += [word_dict]
            elif t == 4:
                word_list += [new_word_dict]
                
            label_list += [label]
        #print(word_list)
            
        f1.close()
            
        with open(str(output_file), "wt") as f2:
            output = ""
            for i in range(len(word_list)):
                output += label_list[i] + "\t"
                for key in word_list[i]:
                    output += key + ":"
                    output += str(word_list[i][key])
                    output += "\t"
                output += "\n"
            f2.write(output) 
            
        print("written!")
      
            
### read the arguments and do operations

'''
if __name__ == "__main__":

    train_input = sys.argv[1]
    valid_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input  = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_valid_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])
    
    #prepare dictionary of words
    dict_out = prepareDict(dict_input)
    
    if feature_flag == 1:
        parseData(dict_out, train_input, formatted_train_out, 1)
        #parseData(dict_out, valid_input, formatted_valid_out, 1)
        #parseData(dict_out, test_input, formatted_test_out, 1)
    if feature_flag == 2:
        parseData(dict_out, train_input, formatted_train_out, 4)
        parseData(dict_out, valid_input, formatted_valid_out, 4)
        parseData(dict_out, test_input, formatted_test_out, 4)
'''

###
dict_out = prepareDict(dict_address)
parseData(dict_out, large_train, "formatted_train_out", 1)
parseData(dict_out, large_valid, "formatted_valid_out", 1)
parseData(dict_out, large_test, "formatted_test_out", 1)
