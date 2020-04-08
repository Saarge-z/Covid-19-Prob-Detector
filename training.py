import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler

def data_split(data, ratio):
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]





if __name__ == "__main__":
    # Read data set
    df = pd.read_csv('data.csv')
    train, test = data_split(df, 0.2)

    X_train = train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
    X_test = test[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()

    Y_train = train[['infectionProb']].to_numpy().reshape(2413,)
    Y_test = test[['infectionProb']].to_numpy().reshape(603,) #according to sklearn reshape(size of data,)


    # Preparations for Feature Scaling
    Xtn,Xtn2 = X_train[:,[0,2]], X_train[:,[1,3,4]]
    Xts,Xts2 = X_test[:,[0,2]], X_test[:,[1,3,4]]


    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(Xtn)
    X_test = sc_X.fit_transform(Xts)

    X_train = np.append(X_train, Xtn2, 1)
    X_test = np.append(X_test, Xts2, 1)
    # After feature Scalling Index is
    # Fever, Age, bodyPain, runnnyNose, diffBreath


    # training model
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)


    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()