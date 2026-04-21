## ML Project 3 - ID3 Algorithm Implementation for Decision Tree Learning

This project implements the ID3 algorithm for creating a decision tree based on a set of training examples. This algorithm was split into several parts, including calculating the entropy of a given attribute, using that entropy to calculate the information gain associated with that attribute, and using this to choose which attribute to add next as a node of the decision tree. The program takes in two arguments, being the name of the input file with the training examples and the name of the output file to print the finished decision tree to. The following is a breakdown of the main files and each of their functions:

main()
This is the main function of the file. It starts by ensuring the right number of arguments, which should include the names of the input and output files to get the data and where to write the final tree. It then reads the input file and gets the examples using the read_data() function and calls the id3 algorithm via the id3() function, passing in the set of examples, the target attribute, and the list of attributes. After the tree is made, the output file is opened and the tree is written there.

read_data(filename)
This is the function called to read the data from the input file. First, all of the lines are read and stored in a list. From this list, the headers are picked out and stored in a separate list. Then, for each of the data lines, a dictionary is being populated with each of the headers and their corresponding data values for each of the training examples. All of these dictionaries are stored in a list and are returned from the function, along with the list of headers.

id3(examples, target_attr, attributes)
This is the function that implements the ID3 algorithm for building a decision tree. It begins by checking one of the base cases where all of the remaining examples are of the same classification by calling the all_same_class() function. If this is the case, then that classification is returned. It then checks the next base case where there are no attributes left, in which case it will return the most common value of the target attribute among the set of given examples. From here, the algorithm will enter the recursive step. It first calculates the information gain that would come from learning each one of the attributes by calling information_gain() and then keeps a variable for the attribute with the best/highest value. It then goes through each possible value for this attribute and creates a subset of the given examples whose value for the target attribute is equal to this current value. If the subset is empty, then the classification for this value of the target attribute in the tree is set to the most common value among the examples using the majority_class() function. Otherwise, the value for this best attribute is set by calling ID3 again, passing in the subset of examples with the current value for this attribute, the same target attribute, and the remaining attributes not including the attribute that was just operated on. In the end, the decision tree is built and returned as a dictionary.

all_same_class(examples, target_attr)
This function goes through each provided example and returns true if all of their classification labels are the same.

majority_value(examples, target_attr)
This function keeps track of the counts for each label in examples and returns the one with the most common/highest count.

information_gain(examples, attr, target_attr)
This function implements the information gain algorithm discussed in class. It first calculates the entropy of the entire set of examples with the entropy() function. Then it goes through each of the values of the attribute in the examples, creates a subset of the examples with the current value, and keeps a sum of the weighted entropy by dividing the size of the subset by the size of the entire set and then multiplying that quotient by the entropy of the subset on the target attribute with the entropy() function. It does this for each of the values, updating the weighted entropy at each value, and in the end returns the value of the old entropy of the entire set minus the new, weighted entropy.

entropy(examples, target_attr)
This function implements the entropy algorithm discussed in class. It starts by getting a count of how many times each classification for the target attributes appears in the examples. Then, for each possible classification, the proportion p is found by dividing the count for the classification by the total number of examples, and then subtracting plog2p from the entropy value. Once it does this for all possible classifications, the entropy is successfully calculated and returned.

## Output Files
PlayTennisOutput.txt : The decision tree generated from the original dataset for PlayTennis for task 2
PlayTennisOutputD4Output.txt : The Decision tree generated after having a modified PLayTennis Dataset for task 4

## How to execute the program

Before starting, make sure the Python interpreter is installed on your machine. Then navigate to the directory where these files are located. Ensure that Python is loaded before attempting to run any of the program files. The following command will ensure Python is loaded:

module load python

To execute the program with your input file, run the following command in the directory where your program and input files are:

python id3.py inputFile.txt outputFile.txt

Where inputFile.txt is the name of the file with your input csv data, and ouputFile.txt is the name of the text file that you want to print the tree output from the program.
