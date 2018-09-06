import json
import csv
import pandas as pd
import Levenshtein
import numpy as np

"""
This assignment can be done in groups of 3 students. Everyone must submit individually.

Write down the UNIs of your group (if applicable)

Member 1: Jinhan Cheng, jc4834

Member 2: Xiangzi Meng, xmm2103
"""

def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
    locu_train = pd.read_json("locu_train.json")
    foursquare_train = pd.read_json("foursquare_train.json")
    matches_train = pd.read_csv("matches_train.csv")
    locu_test = pd.read_json("locu_test.json")
    foursquare_test = pd.read_json("foursquare_test.json")
    # make all strings into lowercase
    locu_train = locu_train.apply(lambda x: x.astype(str).str.lower())
    foursquare_train = foursquare_train.apply(lambda x: x.astype(str).str.lower())
    locu_test = locu_test.apply(lambda x: x.astype(str).str.lower())
    foursquare_test = foursquare_test.apply(lambda x: x.astype(str).str.lower())
    # clean column "website"
    def clean_website(data):    
        for i, row in enumerate(data['website']):
            if not row:
                continue
            else:
                if row.find('.com') > 0:
                    row = row[:row.find('.com')]
                if row.find('www.') > 0:
                    row = row[row.find('www.') + 4:]
                if row.find('//') > 0:
                    row = row[row.find('//') + 2:]
                if row.find('.') > 0:
                    row = row[:row.find('.')]
            data['website'][i] = row
        return data
    locu_train = clean_website(locu_train)
    foursquare_train = clean_website(foursquare_train)
    locu_test = clean_website(locu_test)
    foursquare_test = clean_website(foursquare_test)
    # clean column "phone" and remove punctuation
    def clean_phone(data):
        data['phone'] = data['phone'].str.replace('[^\w\s]','')
        data['phone'] = data['phone'].str.replace(' ','')
        return data
    locu_train = clean_phone(locu_train)
    foursquare_train = clean_phone(foursquare_train)
    locu_test = clean_phone(locu_test)
    foursquare_test = clean_phone(foursquare_test)
    # clean name
    def clean_name(data):
        data['name'] = data['name'].str.replace('[^\w\s]','')
        data['name'] = data['name'].str.replace(' ','')
        return data
    locu_train = clean_name(locu_train)
    foursquare_train = clean_name(foursquare_train)
    locu_test = clean_name(locu_test)
    foursquare_test = clean_name(foursquare_test)
    # clean street address
    def clean_address(data):
        data['street_address'] = data['street_address'].str.replace('[^\w\s]','')
        data['street_address'] = data['street_address'].str.replace(' ','')
        return data
    locu_train = clean_address(locu_train)
    foursquare_train = clean_address(foursquare_train)
    locu_test = clean_address(locu_test)
    foursquare_test = clean_address(foursquare_test)
    # sort
    def sort_foursquare(locu, foursquare):
        foursquare_sort = {'id':[], 'name':[], 'phone':[], 'postal_code':[], 'street_address':[], 'website':[]}       
        N = locu.shape[0]
        Y=list(range(N))
        for i in range(0,N):
            dist2 = []
            for j in range(0,N): 
                dist2.append((float(locu['latitude'][i])-float(foursquare['latitude'][j]))**2+(float(locu['longitude'][i])-float(foursquare['longitude'][j]))**2)   
            Y[i] = dist2.index(min(dist2))                                                               
            foursquare_sort['id'].append(foursquare['id'][Y[i]])
            foursquare_sort['name'].append(foursquare['name'][Y[i]])
            foursquare_sort['phone'].append(foursquare['phone'][Y[i]])
            foursquare_sort['postal_code'].append(foursquare['postal_code'][Y[i]])
            foursquare_sort['street_address'].append(foursquare['street_address'][Y[i]])
            foursquare_sort['website'].append(foursquare['website'][Y[i]])
        foursquare_sort = pd.DataFrame(data=foursquare_sort,columns=['id', 'name', 'phone', 'postal_code', 'street_address', 'website'])
        return foursquare_sort
    # sorted foursquare_test
    foursquare_sort = sort_foursquare(locu_test, foursquare_test)
    # find matches_test
    def predict(locu,foursquare):
        matches = {'locu_id':[],'foursquare_id':[]}
        N = locu.shape[0]
        for i in range(0,N):
            if Levenshtein.distance(locu['name'][i],foursquare['name'][i])==abs(len(locu['name'][i])-len(foursquare['name'][i])):
                matches['locu_id'].append(locu['id'][i])
                matches['foursquare_id'].append(foursquare['id'][i])              
            elif locu['website'][i] != '' and foursquare['website'][i] != '':  
                if locu['website'][i]==foursquare_sort['website'][j]:
                    matches['locu_id'].append(locu['id'][i])
                    matches['foursquare_id'].append(foursquare['id'][i])       
            elif locu['phone'][i] != '' and foursquare['phone'][i] != '':
                if Levenshtein.distance(locu['phone'][i],foursquare['phone'][i]) == abs(len(locu['phone'][i])-len(foursquare['phone'][i])):
                    matches['locu_id'].append(locu['id'][i])
                    matches['foursquare_id'].append(foursquare['id'][i])  
        matches = pd.DataFrame(data=matches,columns=['locu_id','foursquare_id'])
        return matches
    # predict matches_test
    matches_test = predict(locu_test, foursquare_sort)
    # write into csv
    matches_test.to_csv('matches_test.csv',index=False)
    print('done')
    
get_matches("locu_train.json", "foursquare_train.json", "matches_train.csv", "locu_test.json", "foursquare_test.json")

matches_test = pd.read_csv("matches_test.csv")   
    
    
    
    
    
    
    
    
    
    
    
    
    