from keras.applications import MobileNetV2, VGG16, InceptionResNetV2
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras import backend as K
from keras.engine.network import Network
from keras.datasets import fashion_mnist
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf
from sklearn.ensemble import IsolationForest

from data import kyocera_data
from utils.evaluate import caculate_acc
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix



def get_percentile(scores):
        # Threshold was set to 5% for the recall results in the supplements
        #c = anomaly_proportion * 100
        # Highest 5% are anomalous
        per = np.percentile(scores, 100 - 5)
        return per

input_shape = (224, 224, 3)

data_path = 'data'
X_train_s, X_ref, y_ref, X_test_s, X_test_b  = kyocera_data(data_path)

# mobile = MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha,
#                          depth_multiplier=1, weights='imagenet')
mobile = InceptionResNetV2(include_top=False,  input_shape= input_shape, weights='imagenet')


# mobile = VGG16(include_top=True, input_shape=input_shape, weights='imagenet')
mobile.layers.pop() 

model = Model(inputs=mobile.input,outputs=mobile.layers[-1].output)

# flat = GlobalAveragePooling2D()(mobile.layers[-1].output)
# model = Model(inputs=mobile.input,outputs=flat)

model.load_weights('model/model_t_smd_110.h5')

print(X_train_s.shape)
print(X_test_s.shape)
print(X_test_b.shape)

train = model.predict(X_train_s)
test_s = model.predict(X_test_s)
test_b = model.predict(X_test_b)



train = train.reshape((len(X_train_s),-1))
print('reshape train',train.shape)
test_s = test_s.reshape((len(X_test_s),-1))
print('reshape test normal',test_s.shape)
test_b = test_b.reshape((len(X_test_b),-1))
print('reshape test abnormal',test_b.shape)

# train = train.squeeze()
# print('reshape train',train.shape)
# test_s = test_s.squeeze()
# print('reshape test normal',test_s.shape)
# test_b = test_s.squeeze()
# print('reshape test abnormal',test_b.shape)

print('fit model')
# ms = MinMaxScaler()
# train = ms.fit_transform(train)
# test_s = ms.transform(test_s)
# test_b = ms.transform(test_b)

np.save("train.npy",train)
np.save("test_s.npy",test_s)
np.save("test_b.npy",test_b)

# fit the model
# clf = LocalOutlierFactor(n_neighbors=5)
# y_pred = clf.fit(train)

# Z1 = -clf._decision_function(test_s)
# Z2 = -clf._decision_function(test_b)

model2=IsolationForest(contamination='auto',behaviour='new')
print("training...")
model2.fit(train)
print("predicting...")
# y_pred_train = model2.predict(x_train)
# predict_test_s = model2.predict(test_s)
# print("###########")
# print(predict_test_s)
# predict_test_b = model2.predict(test_b)
Z1_score = -model2.decision_function(test_s)
Z2_score = -model2.decision_function(test_b)

per = get_percentile(Z1_score)
Z1 = (Z1_score>=per).astype(int)
 
per = get_percentile(Z2_score)
Z2 = (Z2_score>=per).astype(int) 

#ROC
y_true = np.zeros(len(test_s)+len(test_b))
y_true[len(test_s):] = 1
# path = x_test_s_path + x_test_b_path

# precision, recall, f1 = caculate_acc(y_true, np.hstack((Z1,Z2)),path)

fpr, tpr, _ = metrics.roc_curve(y_true, np.hstack((Z1, Z2)))

# AUC
auc = metrics.auc(fpr, tpr)
print('auc', auc)

cm = confusion_matrix(y_true, np.hstack((Z1, Z2)), labels=None, sample_weight=None)
print('confusion matrix:',cm)
