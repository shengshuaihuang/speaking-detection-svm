#coding utf-8
from sklearn import svm
import pickle
import random


f = open('training.data')
lines = f.readlines()
random.shuffle(lines)

training = []
training_label = []
test = []
test_label = []
trainingNumber = len(lines)

for index,line in enumerate(lines):
	line = [float(val) for val in line.split()]
	if index < trainingNumber:
		training.append(line[:-1])
		training_label.append(line[-1])
	else:
		test.append(line[:-1])
		test_label.append(line[-1])

clf = svm.SVC(class_weight='balanced', C=3)
clf.fit(training, training_label)

# with open('model.pkl', 'wb') as f:
# 	pickle.dump(clf, f)
# f.close()