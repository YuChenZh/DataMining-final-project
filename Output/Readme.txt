1. tempv1 and tempv3 are results of graphlab.item_similarity_recommender.create() 
2. tempv2 is result of matrix factorization, use both uid and qid to make the prediction.
3. tempv4 is result of matrix factorization, use just uid to make the prediction. 
4. tempv5 is result of graphlab.recommender.create()
5. tempv7 is result of matrix factorization, use just qid to make the prediction. 
6. answer is result of hybrid collaborative filtering (user-based plus item-based).