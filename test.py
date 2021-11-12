from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
import tensorflow
import time
import joblib
# 示例数据以minist手写数字为例
# 为了加速计算将原先28X28=784个特征降维至36个
(x_train, y_train),(x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_train = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
print("111")
x_test = (x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print("222")
pca = PCA(n_components=36)
x = pca.fit(train["images"]).transform(train["images"])
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:500]
y_test = y_test[:500]


#定义各算法分类器
NBM = [SVC(kernel='linear', C=0.5),
       SVC(kernel='rbf', C=0.5, gamma='auto'),
       SVC(kernel='poly', C=0.5, degree=3), 
       KNeighborsClassifier(n_neighbors=11), 
       DecisionTreeClassifier(max_depth=15, min_samples_split=5), #防止过拟合将树深设为15
       
NAME= ["LINEAR","RBF","poly", "KNN_N11",, "DCT",]
for itr, itrname in zip(NBM, NAME):
    #训练过程
    print("Training...")
    t1 = time.perf_counter()
    itr.fit(x_train, y_train)
    t2 = time.perf_counter()
    print("Applying...")
    y_train_pdt = itr.predict(x_train)
    t3 = time.perf_counter()
    y_pdt = itr.predict(x_test)
#     joblib.dump(itr, "model/svm_model"+itrname)
    dts1 = len(np.where(y_train_pdt==y_train)[0])/len(y_train)
    dts2 = len(np.where(y_pdt==y_test)[0])/len(y_test)

    print("训练集：{} 精度:{:.3f}%, 训练时间：{:.2f}s".format(itrname, dts1*100, t2 - t1))
    print("测试集：{} 精度:{:.3f}%, 训练时间：{:.2f}s".format(itrname, dts2*100, t3 - t2))
