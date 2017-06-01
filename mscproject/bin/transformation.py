import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, 'mscproject/bin') #so I can call the get_data() function from features.py in this notebook
sys.path.append(os.path.join('mscproject/src')) #for the call to model.utils within features.py


import features
reload(features)

def compute_and_save_transformations(starting_claim_id):
	deficient_claims = []
	deficient_articles = []

	deficient = (-1, -1) #First number is claimId which has <10 articles. Second number is how many articles it actually has
	ending_claim = starting_claim_id+17

	template_csv = pd.read_csv('../../url-versions-2015-06-14-clean-test2.csv')
	my_claims = pd.read_csv('../../my_claims_csv_cleaned.csv')

	all_claims = pd.Series()
	for claimid in range(starting_claim_id, starting_claim_id+18):
	# for claimid in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
	    claims = my_claims.loc[my_claims['claimId']==claimid]
	    if (claims.shape[0]!=10):
	        deficient_claims.append(claimid-starting_claim_id+1)
	        deficient_articles.append(claims.shape[0])
	        deficient = (claimid-starting_claim_id+1, claims.shape[0])
	    claims = claims['claimHeadline']
	    if len(claims) > 10:
	        all_claims = all_claims.append(claims[:10])
	    else:
	        all_claims = all_claims.append(claims)

	all_articles = pd.Series()
	for claimid in range(starting_claim_id, starting_claim_id+18):
	#for claimid in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
	    df = my_claims.loc[my_claims['claimId']==claimid]
	    articles = df['articleHeadline']
	    if len(articles) > 10:
	        all_articles = all_articles.append(articles[:10])
	    else:
	        all_articles = all_articles.append(articles)

	delete_froml = []
	delete_tol = []
	for i in range(len(deficient_claims)):
	    df = 10*(deficient_claims[i]-1) + deficient_articles[i]
	    delete_froml.append(df)
	    dt = 10*(deficient_claims[i])-1
	    delete_tol.append(dt)

	if deficient_claims:
	    for i in range(len(delete_froml)):
	        template_csv = template_csv.drop(range(delete_froml[i], delete_tol[i]+1))

	template_csv['claimHeadline'] = all_claims.values
	template_csv['articleHeadline'] = all_articles.values

	template_csv.to_csv('../data/emergent/Snopes_batch_testing.csv')

	features.get_data(starting_claim_id)

if __name__=='__main__':
	#for i in range(1, 262): #The last batch starts at id 4700, which is (18*261)+2
	compute_and_save_transformations(20)