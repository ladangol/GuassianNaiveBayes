import numpy as np
import pandas as pd

def GuassianPdf(x, m, std):
    term1 = 1/(np.sqrt(2*np.pi)*std)
    terme = (x-m)**2 / (2* (std **2))
    term2 = np.exp(-terme)
    return term1*term2
##########Train and Test Data  dataset3
#traindatafile = 'dataset3_train.txt'
#testdatafile = 'dataset3_test.txt'
#data = np.loadtxt(traindatafile, delimiter="\t")
#testdata = np.loadtxt(testdatafile, delimiter='\t')

################Train and Test Data Dataset1
#datafile = 'dataset1.txt'
#rowdata = np.loadtxt(datafile, delimiter="\t")
#np.random.shuffle(rowdata)
#data= rowdata[:70, :]
#testdata = rowdata[70:, :]
################Train and Test data Dataset2
datafile = 'dataset2.txt'
rowdata = pd.read_csv(datafile, sep="\t", header=None)
rowdata.replace(('Present','Absent'), (1,0), inplace = True)
#data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
#rowdata = np.loadtxt(datafile, dtype = 'str', delimiter="\t")
#rowdata.astype(float)
#rowdata.astype(float)np.array(rowdata.to_records().view(type=np.matrix))
rowdata = rowdata.as_matrix(columns=None)
#rowdata.astype(float)
np.random.shuffle(rowdata)
data= rowdata[:70, :]
testdata = rowdata[70:, :]

#removing the labels
testdatanolabel = testdata[ : , :-1]
true_labels = testdata[:,-1]

row,col = data.shape
class0 = data[np.where(data[:,col-1]==0)]
class1= data[np.where(data[:,col-1]==1)]
 #last column is the label
 #removing the label
class0 = class0[ : , : col-1]
class1= class1[ : , : col-1]

#applying mean and std on each column
#getting a vector of means and standard deviations
mean0 = np.mean(class0, axis = 0)
mean1= np.mean(class1, axis = 0)

std0=np.std(class0,axis =0)
std1=np.std(class1,axis=0)

#calculating the Guassian pdf
pdf_class0 = GuassianPdf(testdatanolabel, mean0, std0)
pdf_class1 = GuassianPdf(testdatanolabel, mean1, std1)

prob0 = np.prod(pdf_class0, axis = 1)  # multiplying the values of each row indinvidually
prob1 = np.prod(pdf_class1, axis = 1)

###Let's predict
#whenever prob1 greater than prob0 so it belongs to class 1
#so True should be 1 otherwise it belongs to class0
result = 1 * (prob1 > prob0)  #1 * is for converting bools to ints
# I could have used (res = prob0 > prob1) and result = res.astype(int) too
#print (result)

### calculating the Accuracy , Precision , Recall measure
TP = np.sum(np.logical_and(result == 1, true_labels ==1))
TN = np.sum(np.logical_and(result==0, true_labels == 0))
FP = np.sum(np.logical_and(result == 1, true_labels == 0))
FN = np.sum(np.logical_and(result == 0, true_labels == 1))

print ('TP: %i, TN: %i, FP: %i, FN:%i' % (TP, TN, FP, FN))
