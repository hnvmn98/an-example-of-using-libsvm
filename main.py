from svmutil import *
from svm import *
import matplotlib.pyplot as plt

y, x = svm_read_problem("train_converted")
py,px = svm_read_problem("test_converted")

m = svm_train(y, x, "-h 0 -c 32 -g 0.5")
#m = svm_train(y, x,"-c 2 -g 0.03125") #train_years 59.125%
#m = svm_train(y, x, "-c 8.0 -g 0.0078125") #train_n 60%
#savemodel = svm_save_model("model",m)
#m = svm_train(y, x, '-c 512.0 -g 3.0517578125e-05') #Accuracy = 59.8972% (466/778)
#m = svm_train(y, x) #feature selected Accuracy = 56.0411%
#m = svm_train(y, x) #Accuracy = 58.6118% (456/778)

p_label, p_acc, p_val = svm_predict(py, px, m)

from sklearn.metrics import f1_score
print f1_score(py, p_label, average='weighted')
