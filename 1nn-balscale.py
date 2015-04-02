# Shashank Juyal
# Roll :201305537
# Assign1: 1 NN classification

import csv
import random
import math
import operator


# Function for finding (first) nearest neighbour
# euclidean distance is used as a distance matrix
def find1NN(trset, test):
	dislist = []
	length = len(test)
	for x in range(len(trset)):
		distance = 0
		# Euclidean Distance with the data present in the test
		
		for j in range(length):
			if j==0: 
				continue
			#print(repr(test[j]))
			distance += pow((test[j] - trset[x][j]), 2)
		dist = math.sqrt(distance)
		dislist.append((trset[x], dist))
	dislist.sort(key=operator.itemgetter(1))
	nearest = []
	nearest.append(dislist[0][0])
	return nearest

# Function to load the data and dividing 
# the data according to the given split value
def init(filename, split, trainset=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for i in range(len(dataset)-1):
	        for j in range(1,5):
	            dataset[i][j] = float(dataset[i][j])
	        if random.random() < split:
	            trainset.append(dataset[i])
	        else:
	            testSet.append(dataset[i])

# Main funtion
def main():
	splitvalues=[0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7]
	sumAccuracy = 0
	acculist = []
	print('\n-------------Running the 1NN classifier-----------\n')
	for i in range(len(splitvalues)):
		trainset=[]
		testSet=[]
		split = splitvalues[i]
		init('balance-scale.data', split, trainset, testSet)
		
		count = 0
		a = {'L': 0, 'B': 0, 'R' : 0}
		b = {'L': 0, 'B': 0, 'R' : 0}
		c = {'L': 0, 'B': 0, 'R' : 0}
		dict = {'L': a, 'B': b, 'R': c}		
		for x in range(len(testSet)):
			nearestneighbor = find1NN(trainset, testSet[x])
			result = nearestneighbor[0][0]
			#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][0]))
			if testSet[x][0] == result:
				count += 1
			dict[result][testSet[x][0]] += 1	
		accuracy = (count/float(len(testSet))) * 100.0
		sumAccuracy += accuracy
		acculist.append(accuracy)
		print('\nRows are : '),
		for key1 in dict:
			print key1,
		print('')
		for key1 in dict:
    			for key2 in dict[key1]:
        			print( repr(dict[key1][key2]) + ' '),
			print('')
		print('Run:' + repr(i+1) + '---- Split:' + repr(split) + ' ---- Accuracy: ' + repr(accuracy) + '%')
	mean = sumAccuracy/len(splitvalues)
	sigma = 0
	#print(acculist)
	for i in range(len(splitvalues)):
		sigma += pow((float(acculist[i])-mean),2)
	stdev = math.sqrt(sigma/len(splitvalues))
	print('\n----------------Final Results----------------')
	print('Mean               : '+ repr(mean) + '\nStandard Deviation : ' + repr(stdev))
	print('Variance           : '+ repr(pow(stdev,2)))
main()



