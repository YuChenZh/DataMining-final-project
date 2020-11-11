import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import data
def dataImport(data_path, title):
    data = pd.read_csv(data_path, sep = ',', skiprows = title, encoding = 'utf-8', names = ['qID', 'uID', 'answer'])
    
    return data

# Train data sort with uID
def sortTrainData(train_data):
    # Omit the title
    raw_data = dataImport(train_data, 1)
    # Differentiate non-answer from non-assign to users. Rescale.
    raw_data.loc[raw_data['answer'] == '1', 'answer'] = '5'
    raw_data.loc[raw_data['answer'] == '0', 'answer'] = '1'
    sort_data = raw_data.sort(['uID'])
    
    return sort_data

# Train data sort with uID
def sortTestData(test_data):
    # Omit the title
    raw_data = dataImport(test_data, 1)
    sort_data = raw_data.sort(['uID'])
    
    return sort_data

# Construct two sparse matrices with train data, one with element question_ID, the 
# other with element answer or not
def sparseMatrixConstruct(data_path):
    raw_data = []  # Combine rows that have the same userID
    ID = []
    question_ID = []
    answer = []
    data_train = sortTrainData(data_path)
    user_ID = '0'
    for i, line in enumerate(data_train):
        l = line.split()
        print(l[2])
        if user_ID == '0':
            user_ID = l[1]
            ID.append(user_ID)
        elif user_ID != l[1]:
            lst1 = [xel1 for xel1 in raw_data[::2]]
            lst2 = [float(xel2) for xel2 in raw_data[1::2]]
            d1 = dict([(m, r) for m, r in zip (lst1, lst1)])
            d2 = dict([(m, r) for m, r in zip (lst1, lst2)])
            question_ID.append(d1)
            answer.append(d2)
            raw_data = []
            user_ID = l[1]
            ID.append(user_ID)
                
        raw_data.append(l[0])
        raw_data.append(l[2])
        lst1 = [xel1 for xel1 in raw_data[::2]]
        lst2 = [float(xel2) for xel2 in raw_data[1::2]]
        d1 = dict([(m, r) for m, r in zip (lst1, lst1)])
        d2 = dict([(m, r) for m, r in zip (lst1, lst2)])
        question_ID.append(d1)
        answer.append(d2)
        
    v1 = DictVectorizer()
    v2 = DictVectorizer(dtype = float)
    question_ID_sparse = v1.fit_transform(question_ID)
    answer_sparse_user = v2.fit_transform(answer)
    answer_sparse_question = np.transpose(v2.fit_transform(answer))
    
    return (question_ID_sparse, answer_sparse_user, answer_sparse_question)

def answerPredict(data_path, question_ID_sparse, answer_sparse_user, answer_sparse_question, neighbor_user_select, neighbor_question_select):
    (x, y, z) = scipy.sparse.find(question_ID_sparse)
    predict_qID = []
    predict_uID = []
    predict_user = []
    predict_question = []
    predict = []
    with sortTestData(data_path) as data_test:
        user_ID = '0'
        u = 0
        for i, line in enumerate(data_test):
            l = line.split()
            predict_qID.append(l[0])
            predict_uID.append(l[1])
            if user_ID == 0:
                user_ID = l[1]
            elif user_ID != l[1]:
                user_ID = l[1]
                u += 1
            
            index = 0
            for j in range(len(z)):
                if z[j] == int(l[0]):
                    index = y[j]
                    break
            similarity_user = cosine_similarity(answer_sparse_user, answer_sparse_user[u])
            similarity_question = cosine_similarity(answer_sparse_question, answer_sparse_question[index])
            sim_user = []
            sim_question = []
            for k_user in range(question_ID_sparse.shape[0]):
                sim_user.append((k_user, similarity_user[k_user]))
            for k_question in range(question_ID_sparse.shape[1]):
                sim_question.append((k_question, similarity_question[k_question]))
            sort_sim_user = sorted(sim_user, key = lambda sim_user: sim_user[1], reverse = True)
            sort_sim_question = sorted(sim_question, key = lambda sim_question: sim_question[1], reverse = True)
            
            k1_user = neighbor_user_select
            k1_question = neighbor_question_select
            k2_user = 0
            k2_question = 0
            sum_user = 0
            sum_question = 0
            for count_user in range(k1_user):
               if answer_sparse_user[sort_sim_user[count_user][0], index] != 0:
                   k2_user += 1
                   sum_user += answer_sparse_user[sort_sim_user[count_user][0], index]
            if k2_user == 0:
                predict_user.append(0)
            else:
                if round(sum_user / k2_user) < 3:
                    predict_user.append(1)
                else:
                    predict_user.append(5)
            for count_question in range(k1_question):
                if answer_sparse_question[sort_sim_question[count_question][0], u] != 0:
                    k2_question += 1
                    sum_question += answer_sparse_question[sort_sim_question[count_question][0], u]
            if k2_question == 0:
                predict_question.append(0)
            else:
                if round(sum_question / k2_question) < 3:
                    predict_question.append(1)
                else:
                    predict_question.append(5)
            
            print(i)
        
        for count in range(len(predict_user)):
            if predict_user[count] == 0:
                predict.append(predict_question[count])
            elif predict_question[count] == 0:
                predict.append(predict_user[count])
            else:
                if round((predict_user[count] + predict_question[count]) / 2) < 3:
                    predict.append(0)
                else:
                    predict.append(1)
                
    return (predict_qID, predict_uID, predict)
    
def resultOutput(output_path, result):
    output = open(output_path, 'w')
    for c_out in range(len(result)):
        output.write(str(result[c_out]) + "\n")
    output.close()
    
if __name__=='__main__':
    (question_ID_sparse, answer_sparse_user, answer_sparse_question) = sparseMatrixConstruct('invited_info_train.txt')
    (result_qID, result_uID, result) = answerPredict('test_nolable.txt', question_ID_sparse, answer_sparse_user, answer_sparse_question, 200, 200)
    resultOutput('answer.csv', (result_qID, result_uID, result))