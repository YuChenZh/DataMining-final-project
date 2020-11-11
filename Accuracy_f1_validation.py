
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

valid_NB =pd.read_table('label_results/naiveBayes_result.txt')
valid_DT =pd.read_table('label_results/decisiontree_result.txt')
valid_RandomF = pd.read_table('label_results/rForest_result.txt')
valid_bayesNetwork = pd.read_table('label_results/bayesNetwork_result.txt')
valid_test =pd.read_table('validate_label2.txt')

######### accuracy-------

acc_nb = accuracy_score(valid_test, valid_NB)
print u'Accuracy of Naive Bayes: %f' % acc_nb

acc_dt = accuracy_score(valid_test, valid_DT)
print u'Accuracy of Decision Trees: %f' % acc_dt

acc_RF = accuracy_score(valid_test, valid_RandomF)
print u'Accuracy of Random Forest: %f' % acc_RF

acc_bn = accuracy_score(valid_test, valid_bayesNetwork)
print u'Accuracy of Bayes Network: %f' % acc_bn


###### f1 score

f1_nb = f1_score(valid_test, valid_NB)
print u'f1 score of Naive Bayes: %f' % f1_nb

f1_dt = f1_score(valid_test, valid_DT)
print u'f1 score of Decision Trees: %f' % f1_dt

f1_rf = f1_score(valid_test, valid_RandomF)
print u'f1 score of Random Forest: %f' % f1_rf

f1_bn = f1_score(valid_test, valid_bayesNetwork)
print u'f1 score of Bayes Network: %f' % f1_bn
