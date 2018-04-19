import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn import over_sampling as os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import keras

# data loading
RANDOM_STATE = 42
df = pd.read_csv('/Users/dyhfr/PycharmProjects/MyProject/DNN/dnntrain.csv')
sub_map= {0:0,1000:1,1500:2,2000:3}
def change(c):
    return sub_map[c]
df['sub'] = df['sub'].apply(change)
df = df.drop('id',axis=1)
df = df.fillna(0)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


X,y = os.SMOTE().fit_sample(X, y)


# build network and classification
def baseline_model():
    model = Sequential()
    model.add(Dense(128, activation='tanh',input_shape=(12,)))
    model.add(Dropout(0.25))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(units=128,activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(units=128,activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
clf = KerasClassifier(build_fn=baseline_model, nb_epoch=40, batch_size=256)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

Y_train_dummy = np_utils.to_categorical(Y_train)
Y_train_dummy.shape

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()
model.add(Dense(128, activation='tanh',input_shape=(12,)))
model.add(Dropout(0.25))
model.add(Dense(units=128, activation='tanh'))
model.add(Dropout(0.25))
model.add(Dense(units=128,activation='tanh'))
model.add(Dropout(0.25))
model.add(Dense(units=128,activation='tanh'))
model.add(Dropout(0.25))
model.add(Dense(4, activation='softmax'))


import keras.backend as K

def precision(y_true, y_pred):
    """Precision metric. Only computes a batch-wise average of precision.
-    Computes the precision, a metric for multi-label classification of
-    how many selected items are relevant.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
-    Only computes a batch-wise average of recall.
-    Computes the recall, a metric for multi-label classification of
-    how many relevant items are selected.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):

    """Computes the F score.

-    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[fmeasure,recall,precision])

history=model.fit(X_train, Y_train_dummy,epochs = 200,
          batch_size=64)

history_dict = history.history
history_dict.keys()

epochs = range(1, len(history_dict['loss']) + 1)


plt.plot(epochs, history_dict['loss'], 'b',label='loss')
plt.plot(epochs, history_dict['fmeasure'], 'r',label='f1')
plt.plot(epochs, history_dict['precision'], 'g',label='precision')
plt.plot(epochs, history_dict['recall'], 'k',label='recall')

plt.xlabel('Epochs')
plt.grid()
plt.legend(loc=1)
plt.show()


Y_pred_class = model.predict_classes(X_test, batch_size=64)
Y_val_class = Y_test
from sklearn import metrics
print("precision", metrics.precision_score(Y_val_class, Y_pred_class, average='weighted'))
print("recall", metrics.recall_score(Y_val_class, Y_pred_class, average='weighted'))
print("f1", metrics.f1_score(Y_val_class, Y_pred_class, average='weighted'))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
from sklearn.metrics import confusion_matrix
import itertools
confusion_mtx = confusion_matrix(Y_val_class, Y_pred_class)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(4))