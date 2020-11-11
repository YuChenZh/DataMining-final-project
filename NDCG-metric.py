import csv
import pandas as pd
from ndcg import ndcg_at_k
import sys,os 

def ndcg4dataset(mapping, ques2user, k):
    def _map_gt(ques, user_score, mapping):
        return [mapping[(ques,user)] for user,score in user_score]

    for ques in ques2user:
        ques2user[ques].sort(key = lambda x: x[1], reverse=True)   

        scores  = [ndcg_at_k(_map_gt(ques, ques2user[ques], mapping), \
                k, method = 1) \
                for ques in ques2user if k <= len(ques2user[ques])]
    evaluated_num = len(scores)
    #print(evaluated_num)
    ndcg_r_score = sum(scores)/evaluated_num

    return ndcg_r_score
	
def NDCG_score(submission_path, truth_path):
    try:
        sub = pd.read_csv(submission_path, encoding='utf-8')
        truth = pd.read_csv(truth_path, encoding='utf-8')
        
        if len(sub['uid']) != len(truth['uid']):
            print("Line number of submission is not correct")
            
        #read truth_file
        mapping = {}
        for record_index,record in truth.iterrows(): 
            qid = record['qid']
            uid = record['uid']
            label = record['label']
            mapping[(qid, uid)] = int(label)

        #read submission_file
        ques2user = {}
        for record_index,record in sub.iterrows(): 
            qid = record['qid']
            uid = record['uid']
            label = record['label']
            if qid not in ques2user:
                ques2user[qid] = []
            ques2user[qid].append((uid, float(label)))  

        ndcg_r_score5 = ndcg4dataset(mapping,ques2user,5)
        ndcg_r_score10 = ndcg4dataset(mapping,ques2user,10)

        score = ndcg_r_score5 * 0.5 + ndcg_r_score10 * 0.5
        print(score)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # data merge without uid and qid
    valid_linReg = 'validationData/linearRegression.csv'
    valid_NB = 'validationData/naivebayes.csv'
    valid_DT ='validationData/decisiontree.csv'
    valid_RandomF = 'validationData/randomforest.csv'
    valid_bayesNetwork = 'validationData/bayesNetwork.csv'
    valid_test ='validate_label.txt'
    NDCG_score(valid_linReg, valid_test)
    NDCG_score(valid_NB, valid_test)
    NDCG_score(valid_DT, valid_test)
    NDCG_score(valid_RandomF, valid_test)
    NDCG_score(valid_bayesNetwork, valid_test)

    # data merge with uid and qid
    valid_linReg_uq = 'validationData/linearRegression-qiduid.csv'
    valid_NB_uq = 'validationData/naivebayes-qiduid.csv'
    valid_DT_uq = 'validationData/decisiontree-qiduid.csv'
    valid_RandomF_uq = 'validationData/randomforest-qiduid.csv'
    valid_bayesNetwork_uq = 'validationData/bayesNetwork-qiduid.csv'
    valid_test = 'validate_label.txt'
    NDCG_score(valid_linReg_uq, valid_test)
    NDCG_score(valid_NB_uq, valid_test)
    NDCG_score(valid_DT_uq, valid_test)
    NDCG_score(valid_RandomF_uq, valid_test)
    NDCG_score(valid_bayesNetwork_uq, valid_test)

    
