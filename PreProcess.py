import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from ucimlrepo import fetch_ucirepo

def load():
    dataset = fetch_ucirepo(id=336)
    ############################################################## ----> manually handle nominal values
    # nodict = {'one':['yes','poor','present','abnormal'], 'zero':['no','good','notpresent','normal']}
    # x = ((dataset.data.features).replace('\t', '', regex=True)).apply(lambda feature: feature.map(lambda value: 1 if value in nodict['one'] else (0 if value in nodict['zero'] else value)))
    ############################################################## ----> auto handle nominal values
    x = ((dataset.data.features).replace('\t', '', regex=True))
    for feature in x.select_dtypes(include=['object']).columns:
        x[feature] = x[feature].astype('category').cat.codes

    y = [1 if i == "ckd" else 0 for i in (dataset.data.targets).replace('\t', '', regex=True)["class"]]

    return np.array(x), np.array(y)

def nan_handler(data):
    ############################################################## ----> to remove samples that has nan
    # x = np.array([list(i) for i in x if not np.isnan(i.any())])
    ############################################################## ----> manually calculate and replace mean
    # mean = []
    # for feature in data.T:
    #     sum = 0
    #     for value in feature:
    #         if not np.isnan(value):
    #             sum += value
    #     mean.append(sum/len(data))
    
    # for i,sample in enumerate(data):
    #     for j, value in enumerate(sample):
    #         if np.isnan(value):
    #             x[i , j] = mean[j]
    ############################################################## ----> auto calculate and replace mean
    data[np.isnan(data)] = np.take(np.nanmean(data, axis=0), np.where(np.isnan(data))[1])
    return x

def numtype(data, threshold= 7):
    data = data.T
    continuous = []
    for i, feature in enumerate(data):
        unique = np.unique(feature)
        if len(unique) > threshold:
            continuous.append(i)

    return continuous

def outlier_detection(data, continuous):
    data = data.T
    for index, feature in enumerate(data):
        if index in continuous:
            sfeature = sorted(feature)
            ############################################################## ----> manually calculate outliers and replace
            # i = len(sfeature)
            # if i % 2 == 0:
            #     q1 = (sfeature[i // 4] + sfeature[i // 4 - 1]) / 2
            #     q3 = (sfeature[3 * i // 4] + sfeature[3 * i // 4 - 1]) / 2
            # else:
            #     q1 = sfeature[i // 4]
            #     q3 = sfeature[3 * i // 4]
            # iqr = q3 - q1

            # lower = q1 - 1.5 * iqr
            # upper = q3 + 1.5 * iqr

            # sum = 0
            # count = 0
            # for i in feature:
            #     if lower < i < upper:
            #         sum += i
            #         count += 1
            # mean = sum / count
            # for i, val in enumerate(feature):
            #     if not lower < val < upper:
            #         feature[i] = mean
            ############################################################## ----> auto calculate outliers and replace
            q1 = np.percentile(sfeature, 25)
            q3 = np.percentile(sfeature, 75)
            iqr = q3 - q1
            
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            outliers = (feature < lower) | (feature > upper)
            inliers = feature[~outliers]
            mean = np.mean(inliers)
            feature[outliers] = mean

            data[index] = feature
    
    return data.T

def discrete_c(data, continuous):
    data = data.T
    for index, feature in enumerate(data):
        if index in continuous:

            n = round((np.mean(feature) - feature.min()) / (feature.max() - feature.min()) * 10)
            if n < 3:
                n = 3

            ############################################################## ----> manually calculate
            # base = feature.min()
            # a = feature.max()
            # threshold = (a - base) / n

            # a = []
            # for bin in range(n):
            #     for i, val in enumerate(feature):
            #         if val <= base + threshold and val not in a:
            #             feature[i] = bin
            #     a.append(bin)
            #     base += threshold
            ############################################################## ----> auto calculate
            width = (feature.max() - feature.min()) / n
            bins = [feature.min() - 1]
            bins.extend([feature.min() + i * width for i in range(1, n)])
            bins.append(feature.max())
            feature = pd.cut(feature, bins= bins, labels= False)

            data[index] = feature
 
    return data.T

def normalization(data):
    ############################################################## ----> min-max is using to normilize ordinal ( all ) features and its correct because ordinals are starting from 0 [ z = (r - 1) / (m - 1) ]
    ############################################################## ----> using loop
    # for i, feature in enumerate(data.T):
    #     for j, value in enumerate(feature):
    #         data.T[i, j] = (value - feature.min()) / (feature.max() - feature.min())
    ##############################################################
    min = data.min(axis=0)
    max = data.max(axis=0)
    return (data - min) / (max - min)

def mixed_distance(data):
    ############################################################## ----> using loop
    # dist = np.zeros((len(data), len(data)))
    # for i in range(len(data)):
    #     for j in range(len(data)):
    #         dist[i,j] = distance.cityblock(data[i,:], data[j,:])
            
    # return dist
    ############################################################## ( data is fully numerical )
    return cdist(data, data, metric='cityblock')

def pr(data):
    for i in data:
        print("\n")
        for j in i:
            print(f"{j:.3f}", end='\t')
            

x, y = load()
x = nan_handler(x)
cont = numtype(x)
x = outlier_detection(x, cont)
x = discrete_c(x, cont)
x = normalization(x)
dist = mixed_distance(x)


print(x)
print("\n")
print(dist)
#pr(x)
#pr(dist)

