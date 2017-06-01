#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:07:31 2017
@author: thanhan
"""

api_key = '3d94496b-9c20-4ae0-abb6-407f8f64c541'

import numpy as np

#from eventregistry import *

def query_er(claim):
    er = EventRegistry(apiKey = api_key)
    q = QueryArticlesIter(conceptUri = er.getConceptUri(claim))
    
    res = []
    for art in q.execQuery(er, sortBy = "date"):
        res.append(art)
        
    return res


def query_er2(claim):
    er = EventRegistry(apiKey = api_key)
    q = QueryArticles()
    # set the date limit of interest
    q.setDateLimit(datetime.date(2014, 4, 16), datetime.date(2014, 4, 28))
    # find articles mentioning the company Apple
    q.addConcept(er.getConceptUri("Apple"))
    # return the list of top 30 articles, including the concepts, categories and article image
    q.addRequestedResult(RequestArticlesInfo(page = 1, count = 30,
        returnInfo = ReturnInfo(articleInfo = ArticleInfoFlags(concepts = True, categories = True, image = True))))
    res = er.execQuery(q)


#pip install --upgrade google-api-python-client
g_api = 'AIzaSyDdcg5mQHVXKWvLIz-pMckpPLYghI3ptbs'

g_id = '000758571849911321468:shcsm3fdnds'

#from googleapiclient.discovery import build




def query_g(claim):
    service = build("customsearch", "v1", developerKey=g_api)

    res = service.cse().list( q=claim, cx=g_id).execute()
        
    return res


import pandas as pd

def process_g(g, claim):
    sources = []
    headlines= []
    
    for i in g['items']:
        source = i['displayLink']
        headline = i['title']
        #res.append((source, headline))
        sources.append(source)
        headlines.append(headline)
    
    n = len(sources)
    
    df = pd.DataFrame({ 'claimHeadline': [claim] * n, \
                        'articleHeadline': headlines, \
                        'claimId': [0] * n, \
                        'articleId': range(n) } )
    
    return (sources, df)

import features

train_data = features.get_dataset('url-versions-2015-06-14-clean-train.csv')
X, y = features.split_data(train_data)
X = features.p.pipeline.fit_transform(X)

def get_features(df):
    xt = features.p.pipeline.transform(df)
    return xt

def get_features_ch(claim, headline):
    df = pd.DataFrame({ 'claimHeadline': [claim], \
                        'articleHeadline': [headline], \
                        'claimId': ['a48c4360-68e6-11e4-b528-9d5aa2d2e8e7'], \
                        'articleId': ['d77bd060-68e6-11e4-b528-9d5aa2d2e8e7'] } )
    
    xt = features.p.pipeline.transform(df)
    return xt

def get_claim_f(dic_s, sources, stances, l = 724):
    f = np.zeros((1, 724))
    for so, st in zip(sources, stances):
        if so not in dic_s: continue
        sid = dic_s[so] - 1 # 1-index to 0-index
        f[0, sid] = st - 1 # 0, 1, 2 to -1, 0, 1
        
    return f

import pickle
def answer(claim, res_g):
    (cmv, dic_s) = pickle.load(open('save_cmv_dics.pkl'))
    
    
    (sources, df) = process_g(res_g, claim)
    xt = get_features(df)
    
    stances = cmv.clf_stance.predict(xt)
    claim_f = get_claim_f(dic_s, sources, stances)
    
    vera = cmv.clf_vera.predict_proba(claim_f)
    
    return (sources, df, vera)
    
    
def gen_res_str(sources, df, vera):
    headlines = df.articleHeadline
    
    res = ""
    for s, h in zip(sources, headlines):
        res = res + s + ': ' + h + '<br>'
    
    pf = int(vera[0][0]* 100)
    pu = int(vera[0][1]* 100)
    pt = int(vera[0][2]* 100)
    res = res + 'Predict veracity: ' + str(pf) + '% False, ' + \
                                    str(pu) + '% Unknown, ' +  \
                                    str(pt) + '% True, '
    
    return res