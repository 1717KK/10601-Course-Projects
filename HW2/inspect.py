#!/usr/bin/env python3
import csv
import math
import sys


##### open and read the file then clean data

if __name__ == '__main__':
  infile = sys.argv[1]
  outfile = sys.argv[2]

with open(str(infile), "rt") as f:
    tsvF = csv.reader(f)
    data = [row for row in tsvF]
    data = data[1:]
    newData = []
    for row in data:
        line = row[0]
        newData += [line.split('\t')[-1]]
    f.close()

##### prepare data for calculation

d = dict()
for element in newData:
    if element not in d:
        d[element] = 1
    else:
        d[element] += 1

allElements = 0
for element in d:
    allElements += d[element]

for element in d:
    d[element] = d[element]/allElements


##### calculate the entropy

entropy = 0

for element in d:
    entropy -= d[element]  * math.log(d[element], 2)  
    
    
##### calculate the error rate

errorRate = 0
largeE = -float(math.inf)

for element in d:
    if float(d[element]) > largeE:
        largeE = float(d[element])

smallE = 1-largeE



##### write out the result

text = "entropy: " + str(entropy) + "\n" + "error: " + str(smallE)

print(text)

with open(str(outfile), "wt") as f:
    f.write(text)
    
print("written!")

    
