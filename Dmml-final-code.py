import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import Birch
from sklearn.decomposition import PCA

np.random.seed(0)



data = pd.read_csv(r"C:\Users\ikpri\Downloads\data_tr.txt",header=None,sep='\t')



test_data = pd.read_csv(r"C:\Users\ikpri\Downloads\data_ts.txt",header=None,sep='\t')





def Preprocessing(train_data,test_data):
    
    scaler=StandardScaler()
    X_train= pd.DataFrame(train_data)
    X_test= pd.DataFrame(test_data)
    var_thr = VarianceThreshold(threshold = 0.005)
    var_thr.fit(X_train)
    constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[var_thr.get_support()]]
    print(len(constant_columns))
    df_train = X_train.drop(constant_columns,axis=1)
    df_test = X_test.drop(constant_columns,axis=1)
    
    
    
    data_train_scaled =  scaler.fit_transform(df_train)
    data_test_scaled =  scaler.transform(df_test)
    
    pca = PCA(n_components=50)
    data_train_scaled_pca = pca.fit_transform(data_train_scaled)
    data_test_scaled_pca = pca.transform(data_test_scaled)
    
    return data_train_scaled_pca,data_test_scaled_pca



def mycluster(train,test):
    
    train_data,test_data = Preprocessing(train,test)

    cluster = Birch(n_clusters=16).fit(train_data)
    pred = cluster.predict(test_data)
    score  = metrics.silhouette_score(test,pred)

    print(score)
    
    return pred





def score_api(pred):
    import requests
    import json
    from json import JSONEncoder
    import numpy

    url = "https://www.csci555competition.online/scoretest"

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, numpy.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    payload = json.dumps(pred,cls=NumpyArrayEncoder)

    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)




predicted_clusters = mycluster(data,test_data)

score_api(predicted_clusters)

    
    
    
    
    
    
    
    
    




    
    
    
    
    


    
    
    