from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn import over_sampling as os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

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

#data oversampling
X,y = os.SMOTE().fit_sample(X, y)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = SVC(kernel='rbf',max_iter=200,random_state=RANDOM_STATE)
clf.fit(X_train,Y_train)
predict = clf.predict(X_test)
print(classification_report_imbalanced(Y_test,predict))
print(confusion_matrix(Y_test,predict))

def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title = 'confusion matrix',
                          cmap = plt.cm.Blues):
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

confusion_mtx_svm = confusion_matrix(Y_test,predict)
plot_confusion_matrix(confusion_mtx_svm, classes=range(4))