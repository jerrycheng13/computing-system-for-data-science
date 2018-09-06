
# coding: utf-8

# In[1]:


import os
import random
from operator import *
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import *
conf = SparkConf("local").setAppName("App")
sc = SparkContext(conf = conf)
os.getcwd()


# In[2]:


def artistsDataPassing(line):
    tokens = line.split('\t')
    if len(tokens) < 2: return []
    else:
        try:
            Id = int(tokens[0])
            name = tokens[1].strip()
            return [(Id, name)]
        except:
            return []


# In[3]:


inputData = sc.textFile('artist_data.txt')
artistData = inputData.flatMap(artistsDataPassing)


# In[4]:


def parser(s, delimeters=" ", to_int=None):
    s = s.split(delimeters)
    if s[0]:
        if to_int:
            return tuple([int(s[i]) if i in to_int else s[i] for i in range(len(s))])
        return tuple(s)
    else:
        return tuple([0,0])


# In[5]:


artistAlias = sc.textFile("artist_alias.txt").map(lambda x: parser(x,'\t', [0,1]))
artistAliasMap = artistAlias.collectAsMap()
del artistAliasMap[0]
userArtistData = sc.textFile("user_artist_data.txt").map(lambda x: parser(x,' ',[0,1,2]))
userArtistData = userArtistData.map(lambda x: (x[0], artistAliasMap.get(x[1], x[1]), x[2]))


# In[6]:


artistData.take(10)


# In[7]:


def summary(user_id):
    play_list = userArtistData.map(lambda x: (x[0], (x[1], x[2]))).lookup(user_id)
    total = sum(x[1] for x in play_list)
    print("User %s has a total play count of %s and a mean play count of %s." % (user_id, total, total/len(play_list)))


# In[7]:


summary(2093760)


# ## Local Run

# In[8]:


traindata, validationdata, testdata = userArtistData.randomSplit([15,15,70], 13)
traindata.cache()
testdata.cache()
validationdata.cache()


# In[9]:


training = traindata.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))
model = ALS.trainImplicit(training, rank=10, seed=345)


# In[10]:


recommendation = model.recommendProducts(2093760, 10)
for r in recommendation:
    print(r)


# In[11]:


from functools import reduce
recommendedProductIDs = set(map(lambda x: x.product, recommendation))
for artist in artistData.filter(lambda x: x[0] in recommendedProductIDs).values().collect():
    print(artist)


# ## AWS Run

# In[ ]:


traindata, validationdata, testdata = userArtistData.randomSplit([40,40,20], 13)
traindata.cache()
testdata.cache()
validationdata.cache()
print(traindata.take(3))
print(testdata.take(3))
print(validationdata.take(3))
print(traindata.count())
print(testdata.count())
print(validationdata.count())


# In[8]:


#from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

#def cal_score(predict, actual):
#    if len(actual) < len(predict):
#        predict = predict[0:len(actual)]
#    return len(list(set(predict) & set(actual)))*1.0/len(actual)

#def modelEval(model, dataset):
    # Find the list of all artists in the whole data set
#    all_artists = userArtistData.map(lambda x: x[1]).distinct().collect()
    # Find the users in the input dataset
#    test_user = dataset.map(lambda p: p[0]).distinct().collect()
    # Find the artists each user listened to in the training set and generate the test data
#    global traindata
#    testdata = traindata.filter(lambda x: x[0] in test_user).map(lambda x: (x[0], x[1])).groupByKey()
#    testdata = testdata.map(lambda x: (x[0], list(x[1])))
#    testdata = testdata.flatMap(lambda x: [(x[0],a) for a in all_artists if a not in x[1]])
    # Find the artists each user listened to in the input dataset
#    testdata_actual = dataset.map(lambda x: (x[0], x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))).collectAsMap()
#    predictions = model.predictAll(testdata).map(lambda x: (x[0], (x[1], x[2])))
#    predictions = predictions.groupByKey().map(lambda x: (x[0], sorted(list(x[1]), key=lambda y: y[1], reverse=True)))
#    predictions = predictions.map(lambda x: (x[0], cal_score([y[0] for y in x[1]], testdata_actual[x[0]])))
#    return predictions.map(lambda x:x[1]).reduce(lambda x, y: x+ y) * 1.0 / len(test_user)


# In[11]:


training = traindata.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))
for r in [5, 10, 20]:
    model = ALS.trainImplicit(training, rank = r, iterations = 5, seed=345)
    print("The model score for rank %s is %s" % (r, modelEval(model, validationdata)))

