#   CS365 : Lab E : K-Nearest-Neighbor Classifier
#
#   4-27-17
#
#   Porter Libby
#   Sowwma Roy

import math # for calculating euclideanDist
import csv  # for importing ls files
import random # for shuffling examples to split
import os   # for clearing terminal display
import sys  # for getting args from command line

def loadFile(url):      # load an external file and break it into structured examples, return as list
    with open(url) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')    #open file, breaking elems over ','
        exs_out = []

        for row in readCSV:     #split each row into an example in the form [class, atr, atr, atr, atr...]
            example = []
            for attr in row:
                example.append(attr)
            exs_out.append(example)
        return exs_out

def outputFile(lines, name):    # write confusion matrix to results file
    f = open(name,'w')
    for line in lines:
        row = ''
        for word in line:
            row += str(word) + ','
        f.write(row + '\n')
    f.close()

def isfloat(value): #check if a value can be a float, or if it contains non-number ascii chars
  try:
    float(value)
    return True
  except ValueError:
    return False

def euclideanDist(ex1, ex2, l):       # returns the euclidean distance between two examples of the form [atr,atr,atr...]. l = length
    dist = 0
    for x in range(l):
        if isfloat(ex1[1]):     #if value is a floating point, then use it normally, if not, treat it as yes/no
            dist += pow((float(ex1[x]) - float(ex2[x])), 2)
        else:
            if ex1[x] != ex2[x]:
                dist += 1
    return math.sqrt(dist)

def mode(ls):
    lst = []
    hgh = 0
    for i in range(len(ls)):
        lst.append(ls.count(ls[i]))
    m = max(lst)
    ml = [x for x in ls if ls.count(x) == m ] # to find most frequent values
    mode = []
    for x in ml: # to remove duplicates of mode
        if x not in mode:
            mode.append(x)
    return mode

def splitTestTrain(examples, percent, seed):      # use a seed and a percent to split example ls in to training and test sets
    train_count = float(len(examples)) * percent

    random.Random(seed).shuffle(examples)      #shuffle examples based on seed

    test_examples = []
    train_examples = []

    for x in range(len(examples)):      #split examples based on percentage
        if x < train_count - 1:
            train_examples.append(examples[x])
        else:
            test_examples.append(examples[x])

    return test_examples,train_examples

def kNearestNeighbor(target_example, train_examples,k):
    exs_length = len(target_example) - 1 # number of attributes in each example
    distance_list = []  # list of the euclideanDist of each train_example from the target_example

    for example in train_examples:
        dist = euclideanDist(example[1:], target_example[1:], exs_length)
        distance_list.append(dist)

    sorted_train_examples = [ex for _, ex in sorted(zip(distance_list, train_examples))]    # sort train_examples, based on values from distance_list

    class_guesses = []  # list to hold candidate class guesses from KNN
    for x in range(k):
        class_guesses.append(sorted_train_examples[x][0])
    if len(class_guesses) > 1:
        guess = mode(class_guesses) # get most common guess from list, if there is a tie, the first to be found will be used.
    else:
        guess = class_guesses

    return target_example[0],guess[0]

def print_loading_scrn(point_x,point_y):    # purely aesthetic. displays current progress of calculation
    os.system('cls||clear')
    percent = round(float(point_x) / float(point_y),4)

    part1 = '#'*int(percent*21)
    part2 = '-'*int(21 - (percent*21))
    print('loading...')
    print(str(round((percent*100),2)) + '%, (classifying example ' + str(point_x) + ')')
    print('|' + part1 + part2 + '|')

def generateConfusionMatrix(examples, guess_answer_pairs, attr_titles): # generate confusion matrix
    classes = []    #create matrix form using 0 as value placeholder
    for example in examples:
        if example[0] not in classes:
            classes.append(example[0])
    matrix = [classes]
    for x in range(len(classes)):
        row = []
        for y in range(len(classes)):
            row.append(0)
        row.append(classes[x])
        matrix.append(row)

    for pair in guess_answer_pairs:     # fill in values counting from guess/answer pairs.
        guess = pair[0]
        answer = pair[1]
        x_coord = matrix[0].index(guess)
        y_coord = matrix[0].index(answer) + 1
        matrix[y_coord][x_coord] = matrix[y_coord][x_coord] + 1
    return matrix


def main(url, percent, seed, k):     # control wrapper
    examples = loadFile(url)        # get examples list with labels attached
    attr_titles = examples.pop(0)       # extract labels

    test_examples,train_examples = splitTestTrain(examples,percent,seed)    # split data

    guess_answer_pairs = []     #to hold guess and solution pairs [class guess, actual class]

    for t in range(len(test_examples)):
        actual_class, guess_class = kNearestNeighbor(test_examples[t], train_examples, k)   # use KNN to determine a guess
        guess_answer_pairs.append([guess_class,actual_class])
        print_loading_scrn(t,len(test_examples))

    os.system('cls||clear')     #clear screen for result print
    correct_guesses = 0

    for pair in guess_answer_pairs:
        #print('guess: ' + str(pair[0]) + ' | actual: ' + str(pair[1]))
        if pair[0] == pair[1]:
            correct_guesses += 1

    p_correct = (float(correct_guesses) / float(len(guess_answer_pairs))) * 100
    matrix = generateConfusionMatrix(examples,guess_answer_pairs,attr_titles[1:])

    print('K = ' + str(k))
    print('test accuracy: ' + str(round(p_correct,2)) + '%')
    print(str(correct_guesses) + '/' + str(len(guess_answer_pairs)) + ' examples classified correctly.')

    name = 'results_' + str(url[:-4]) + '_' + str(k) + '_' + str(seed) + '.csv'

    outputFile(matrix,name)
    print('Confusion matrix written to ' + name)


if __name__ == "__main__":      # collect input from terminal user
    # EXAMPLE:    main('mnist_large.csv', 0.75, 12345, 3)
    url = str(sys.argv[1])
    percent = float(sys.argv[2])
    seed = int(sys.argv[3])
    k = int(sys.argv[4])
    main(url,percent,seed,k)
