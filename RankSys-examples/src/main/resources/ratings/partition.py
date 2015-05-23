import random, math

myfile = 'ratings.dat'
myseed=11109
k=5


# Load data
data = open(myfile).readlines()

# output files
train={}
test={}

for i in range(k):
	test[i] = open('test.{}'.format(i), 'w')
	train[i] = open('train.{}'.format(i), 'w') 
	
# Shuffle input
random.seed=myseed
random.shuffle(data)

# Compute partition size given input k
len_part=int(math.ceil(len(data)/float(k)))

# Create one partition per fold
for ii in range(k):
	testdata = data[ii*len_part:ii*len_part+len_part]
	test[ii].write(''.join(str(line) for line in testdata))
	trainingdata = [jj for jj in data if jj not in test]
	train[ii].write(''.join(str(line) for line in trainingdata))
	
#close files
for i in range(k):
	test[i].close()
	train[i].close() 