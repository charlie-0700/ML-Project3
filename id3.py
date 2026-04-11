from collections import Counter
import math
import sys


# Compute entropy of the dataset for the target attribute
def entropy(examples, target_attr):
    total = len(examples)

    # Count how many times each class appears
    counts = Counter(example[target_attr] for example in examples)
    entropy_value = 0
    for count in counts.values():
        p = count / total  # The probability of this particular class
        if p > 0:
            entropy_value -= p * math.log2(p)  # calculate entropy
    return entropy_value


# Compute the information gain of splitting on attribute attr
def information_gain(examples, attr, target_attr):
    total_entropy = entropy(examples, target_attr)
    total = len(examples)

    # Get all of the unique values of the attribute
    values = set(example[attr] for example in examples)

    weighted_entropy = 0

    # Compute weighted entropy after split
    for value in values:
        subset = [example for example in examples if example[attr] == value]
        weighted_entropy += (len(subset) / total) * entropy(subset, target_attr)

    # Information Gain = subtract the new entropy from the original entropy
    return total_entropy - weighted_entropy


# Return the most common class label
def majority_value(examples, target_attr):
    counts = Counter(example[target_attr] for example in examples)
    return counts.most_common(1)[0][0]


# Check if all examples have the same class label
def all_same_class(examples, target_attr):
    first = examples[0][target_attr]
    return all(example[target_attr] == first for example in examples)


# The actual ID3 algorithm
def id3(examples, target_attr, attributes):

    # Situation 1 - all of the examples are in the same class, so return that class
    if all_same_class(examples, target_attr):
        return examples[0][target_attr]

    # Situation 2 - no attributes are left, return the majority class
    if not attributes:
        return majority_value(examples, target_attr)

    # Choose the attribute with the highest IG
    best_attr = None
    best_gain = -1

    for attr in attributes:
        gain = information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    # Create the tree with the best attribute as the root
    tree = {best_attr: {}}

    # Get all of the possible values of the best attribute
    values = set(example[best_attr] for example in examples)

    for value in values:
        subset = [example for example in examples if example[best_attr] == value]

        # If subset is empty, use the majority class
        if not subset:
            tree[best_attr][value] = majority_value(examples, target_attr)
        else:
            remaining_attrs = [attr for attr in attributes if attr != best_attr]
            tree[best_attr][value] = id3(subset, target_attr, remaining_attrs)
    return tree


# functionn to read in data
def read_data(filename):
    examples = []
    with open(filename, "r") as file:
        lines = file.readlines()

    headers = lines[0].strip().split(",")

    for line in lines[1:]:
        values = line.strip().split(",")
        row = {}
        for i in range(len(headers)):
            row[headers[i]] = values[i]
        examples.append(row)
    return examples


# main program

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 id3.py <training_file>")
        sys.exit(1)

    filename = sys.argv[1]
    examples = read_data(filename)

    target_attr = "PlayTennis"
    attributes = ["Outlook", "Temperature", "Humidity", "Wind"]

    tree = id3(examples, target_attr, attributes)
    print(tree)
