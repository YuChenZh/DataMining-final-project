# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


# load data, item_id is qid, user_id is uid
in_train = pd.read_table('invited_info_train.txt', sep='\t')
in_train.columns = ["qid","uid","label"]

qu_info = pd.read_table('question_info.txt', sep='\t')
qu_info.columns = ["qid","qtag","u_decrip_x","u_decrip2_x","num_upvotes","num_anw","num_topanw"]

user_info = pd.read_table('user_info.txt', sep='\t')
user_info.columns = ["uid","utag","u_decrip_y","u_decrip2_y"]

vali_data = pd.read_table('validate_nolabel.txt', sep=',')

#### Seperate the label column for accuracy metric and f1 score metric
# vali_label = pd.read_table('validate_label.txt', sep=',')
# vali_label2=vali_label['label']
#
# vali_label2.to_csv('validate_label2.txt',index=False)


# ########################  train data preprocess ##############################

# join files into train
join_tr1 = pd.merge(in_train, qu_info, on='qid', how='left')
join_tr2 = pd.merge(join_tr1, user_info, on='uid', how='left')

# to string, for tf-idf use
join_tr2['qtag']=join_tr2['qtag'].astype(str)
join_tr2['u_decrip']=join_tr2['u_decrip_x'].astype(str)
join_tr2['u_decrip2']=join_tr2['u_decrip2_x'].astype(str)
join_tr2['num_upvotes']=join_tr2['num_upvotes'].astype(str).str.replace('\[|\]|\'', '')
join_tr2['num_topanw']=join_tr2['num_topanw'].astype(str).str.replace('\[|\]|\'', '')
join_tr2['num_anw']=join_tr2['num_anw'].astype(str).str.replace('\[|\]|\'', '')

join_tr2['utag']=join_tr2['utag'].astype(str).str.replace('\[|\]|\'', '')

join_tr2['combined'] = join_tr2['uid'].map(str) +','+ join_tr2['qid'].map(str) +','+ join_tr2['qtag'].map(str) +','+ join_tr2['u_decrip'].map(str)+','+join_tr2['u_decrip2'].map(str)+','+join_tr2['num_upvotes'].map(str)+','+join_tr2['num_topanw'].map(str)+','+join_tr2['num_anw'].map(str)+','+join_tr2['utag']

# ########################  test data preprocess ##############################
# join files into train
join_te1 = pd.merge(vali_data, qu_info, on='qid', how='left')
join_te2 = pd.merge(join_te1, user_info, on='uid', how='left')

# to string, for tf-idf use
join_te2['qtag']=join_te2['qtag'].astype(str).str.replace('\[|\]|\'', '')
join_te2['u_decrip']=join_te2['u_decrip_y'].astype(str).str.replace('\[|\]|\'', '')
join_te2['u_decrip2']=join_te2['u_decrip2_y'].astype(str).str.replace('\[|\]|\'', '')
join_te2['num_upvotes']=join_te2['num_upvotes'].astype(str).str.replace('\[|\]|\'', '')
join_te2['num_topanw']=join_te2['num_topanw'].astype(str).str.replace('\[|\]|\'', '')
join_te2['num_anw']=join_te2['num_anw'].astype(str).str.replace('\[|\]|\'', '')

join_te2['utag']=join_te2['utag'].astype(str).str.replace('\[|\]|\'', '')

join_te2['combined'] = join_tr2['uid'].map(str) +','+ join_tr2['qid'].map(str) +','+ join_te2['qtag'].map(str) +','+ join_te2['u_decrip'].map(str)+','+join_te2['u_decrip2'].map(str)+','+join_te2['num_upvotes'].map(str)+','+join_te2['num_topanw'].map(str)+','+join_te2['num_anw'].map(str)+','+join_te2['utag']


# tf-idf
tr_joined = join_tr2['combined']
te_joined = join_te2['combined']
v = TfidfVectorizer()
train_data = v.fit_transform(tr_joined)
test_data = v.transform(te_joined)

# train label
train_label = np.asarray(join_tr1['label'])

######---- Linear Regression------########
lr = linear_model.LinearRegression()
lr.fit(train_data, train_label)

predicted = lr.predict(test_data)

# Output result - sim_model
out_lr = open('label_results/regression_predict.csv','w')
maxv_lr = max(predicted)
minv_lr= min(predicted)
for n in range(0, len(predicted)):
    x= predicted[n]
    out_lr.write(str((x - minv_lr)/(maxv_lr - minv_lr)) + "\n")
out_lr.close()


######---- Decision tree------########
clf_dt = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=2, splitter='best')
clf_dt.fit(train_data, train_label)
dt_predict=clf_dt.predict(test_data)

# Output result
out_dt = open('label_results/decisiontree_result.txt','w')
for n in range(0, len(dt_predict)):
    out_dt.write(str(dt_predict[n]) + "\n")
out_dt.close()

# #### ------- RandomForestClassifier-------

clf_rf = RandomForestClassifier()
clf_rf.fit(train_data, train_label)
rf_predict=clf_rf.predict(test_data)

# Output result
out_rf = open('label_results/rForest_result.txt','w')
for n in range(0, len(rf_predict)):
    out_rf.write(str(rf_predict[n]) + "\n")
out_rf.close()


#### -------Bayesian Network, Bernoulli event model

clf_bn = BernoulliNB()
clf_bn.fit(train_data, train_label)
bn_predict = clf_bn.predict(test_data)

# Output result
out_bn = open('label_results/bayesNetwork_result.txt','w')
for n in range(0, len(bn_predict)):
    out_bn.write(str(bn_predict[n]) + "\n")
out_bn.close()

#### ------- naive bayes-------

clf_nb = MultinomialNB()
clf_nb.fit(train_data, train_label)
nb_predict=clf_nb.predict(test_data)

# Output result
out_naB = open('label_results/naiveBayes_result.txt','w')
for n in range(0, len(nb_predict)):
    out_naB.write(str(nb_predict[n]) + "\n")
out_naB.close()


