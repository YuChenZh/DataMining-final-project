# -*- coding: utf-8 -*-

import pandas as pd
import graphlab

# load data, item_id is qid, user_id is uid
in_train = pd.read_table('invited_info_train.txt', sep='\t')
in_train.columns = ["qid","uid","label"]

qu_info = pd.read_table('question_info.txt', sep='\t')
qu_info.columns = ["qid","qtag","u_decrip","u_decrip2","num_upvotes","num_anw","num_topanw"]

user_info = pd.read_table('user_info.txt', sep='\t')
user_info.columns = ["uid","utag","u_decrip","u_decrip2"]

vali_data = pd.read_table('validate_nolabel.txt', sep=',')
vali_data.to_csv('validate_nolabel.csv')

test_data = pd.read_table('test_nolabel.txt', sep=',')
test_data.to_csv('test_nolabel.csv')

# print vali_data[0:100]

train_dataS = graphlab.SFrame(in_train)
test_dataS = graphlab.SFrame(vali_data)
user_infoS = graphlab.SFrame(user_info)
qu_infoS = graphlab.SFrame(qu_info)

####### ----- item similarity models-------

#Train Model
sim_model = graphlab.item_similarity_recommender.create(train_dataS, user_id='uid', item_id='qid', target='label', similarity_type='pearson')
#Make Recommendations
sim_predic = sim_model.predict(test_dataS)


# Output result - sim_model
out_sim = open('label_results/label_predict.csv','w')
for n in range(0, len(sim_predic)):
    out_sim.write(str(sim_predic[n]) + "\n")
out_sim.close()

####### ----- factorization recommenders-------

# use user_id and item_id
fac_model_uq = graphlab.factorization_recommender.create(train_dataS, target='label',
                                                       user_id='uid', item_id='qid',
                                                       user_data=user_infoS,
                                                       item_data=qu_infoS)

fac_predict_uq = fac_model_uq.predict(test_dataS, new_user_data=user_infoS, new_item_data=qu_infoS)

# Output result - sim_model
out_fac_uq = open('label_results/fac_predict_uq.csv','w')
maxv_uq = max(fac_predict_uq)
minv_uq= min(fac_predict_uq)
for n in range(0, len(fac_predict_uq)):
    x= fac_predict_uq[n]
    out_fac_uq.write(str((x - minv_uq)/(maxv_uq - minv_uq)) + "\n")
out_fac_uq.close()

# use user_id(uid)
sim_model_u = graphlab.factorization_recommender.create(train_dataS, target='label',
                                                       user_id='uid', item_id='qid',
                                                       user_data=user_infoS)

sim_predict_u = sim_model_u.predict(test_dataS, new_user_data=user_infoS)

# Output result - sim_model
out_sim_u = open('label_results/fac_predict_u.csv','w')
maxv_u = max(sim_predict_u)
minv_u= min(sim_predict_u)
for n in range(0, len(sim_predict_u)):
    x= sim_predict_u[n]
    out_sim_u.write(str((x - minv_u)/(maxv_u - minv_u)) + "\n")
out_sim_u.close()


# use item_id(qid)
sim_model_q = graphlab.factorization_recommender.create(train_dataS, target='label',
                                                        user_id='uid', item_id='qid',
                                                        item_data=qu_infoS)

sim_predict_q = sim_model_q.predict(test_dataS, new_item_data=qu_infoS)

# Output result - sim_model
out_sim_q = open('label_results/fac_predict_q.csv','w')
maxv_q = max(sim_predict_q)
minv_q= min(sim_predict_q)
for n in range(0, len(sim_predict_q)):
    x= sim_predict_q[n]
    out_sim_q.write(str((x - minv_q)/(maxv_q - minv_q)) + "\n")
out_sim_q.close()

####### ----- recommenders-------


model = graphlab.recommender.create(train_dataS, target='label', user_id='uid', item_id='qid')
predict = model.predict(test_dataS)

# Output result -
out = open('label_results/label.csv','w')
maxv = max(predict)
minv= min(predict)
for n in range(0, len(predict)):
        x2= predict[n]
        out.write(str((x2 - minv)/(maxv - minv)) + "\n")
out.close()


####### ----- recommenders-final test-------

test_dataF = graphlab.SFrame(test_data)

model_F = graphlab.recommender.create(train_dataS, target='label', user_id='uid', item_id='qid')
predict_F = model_F.predict(test_dataF)

# Output result - test
out_F = open('label_results/label_test.csv','w')
maxv_F = max(predict_F)
minv_F= min(predict_F)
for n in range(0, len(predict_F)):
        x2= predict_F[n]
        out_F.write(str((x2 - minv_F)/(maxv_F - minv_F)) + "\n")
out_F.close()
