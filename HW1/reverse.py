# Note: As this requires read-write access to your hard drive,
#       this will not run in the browser in Brython.
import os



with open("example.txt", "rt") as f:
    result = []
    for line in f.readlines():
        result += [line]
    result.reverse()
    string = ""
    for line in result:
        string += line
        
with open("output.txt", "wt") as f:
    f.write(string)
    
print("written!")
    