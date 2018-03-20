#coding utf-8
from sklearn import svm
import numpy as np
import pickle
import random
import sys, getopt

inputFile = ''
outputFile = ''

#打开test文件
if len(sys.argv)==2:
	inputFile = sys.argv[1]
elif len(sys.argv) == 1:
	inputFile = 'test.data'
else:
	print('Please type test filename behind the py file!')
 
f = open(inputFile,'rb')
lines = f.readlines()
f.close()

#载入分类模型
with open('model.pkl', 'rb') as modelFile:
	clf = pickle.load(modelFile)
modelFile.close()

#载入分类模型
predY = []
groundTruth = []
for line in lines:
	line = [float(val) for val in line.split()]
	predY.append(clf.predict([line[:6]])[0])		#对单个样本进行预测
	try:
		groundTruth.append(line[6])		#如果有标签的话
	except:
		pass

#如果test有标签，则进行比较计算precision
if len(groundTruth) != 0:
	predY_array = np.array(predY)
	groundTruth_array = np.array(groundTruth)
	precision = (len(groundTruth) + np.dot(predY_array, groundTruth_array))/(2*len(groundTruth))
	print("precision is %s" % precision)

#保存预测结果到result.txt
sf = open('result.txt','wb')
for predy in predY:
	sf.write(str(int(predy)).encode())
	sf.write("\r\n".encode())
sf.close()
print("predY saved in result.txt")
