from sklearn.datasets import fetch_openml
from sklearn import svm
import joblib
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")
#以下为用opencv读取我手写的图片
img5=cv2.imread("../5.png")
imgr5=cv2.resize(img5,(28,28),interpolation= cv2.INTER_LINEAR)
imgr5=cv2.split(imgr5)
imgr5=imgr5[0]

img0=cv2.imread("../0.png")
imgr0=cv2.resize(img0,(28,28),interpolation= cv2.INTER_LINEAR)
imgr0=cv2.split(imgr0)
imgr0=imgr0[0]

img3=cv2.imread("../3.png")
imgr3=cv2.resize(img3,(28,28),interpolation= cv2.INTER_LINEAR)
imgr3=cv2.split(imgr3)
imgr3=imgr3[0]

load=False
#lr是一个LogisticRegression模型
#是否加载训练集
if(load):
    mnist = fetch_openml("mnist_784")
    data = mnist['data'][:2500]
    label = mnist['target'][:2500]
    test_data=mnist['data'][:-500]
    test_label = mnist['target'][:-500]
    print(type(data))
    print(len(data))
    print(test_data[:1])
#线性平面分割
model_linear = svm.SVC(kernel='linear', C = 0.001,decision_function_shape='ovr')
#是否训练
train=False
if(train and load):
    # 训练线性内核
    print("start training lr")
    model_linear.fit(data, label)  # 训练
    accuracy = model_linear.score(test_data, test_label)#训练打分
    print(accuracy)
    joblib.dump(model_linear, 'lr.model')#joblib保存训练模型
    print("save lr success")

model_linear=joblib.load('lr.model')#joblib读取训练好的模型

#预测5，0，3 三个数字
cv2.imshow("resized_5",imgr5)
pre_5=imgr5.reshape(1,784)
print(f"预测数字5={model_linear.predict(pre_5)}")

cv2.imshow("resized_0",imgr0)
pre_0=imgr0.reshape(1,784)
print(f"预测数字0={model_linear.predict(pre_0)}")

cv2.imshow("resized_3",imgr3)
pre_3=imgr3.reshape(1,784)
print(f"预测数字3={model_linear.predict(pre_3)}")
#可以看到503预测正确
cv2.waitKey()

"""
#训练sigmoid内核 结果与下两种相似
model_poly=svm.SVC(kernel='sigmoid',C=0.001,gamma=0.05,decision_function_shape='ovr')
#训练多项式内核
print("start training sd")
model_poly.fit(data, label) # 训练
accuracy=model_poly.score(test_data,test_label)
print(accuracy)
joblib.dump(model_poly, 'sd.model')
print("save sd success")
#通过训练可以看出非线性核带来的效果很差0.09746762589928057

model_poly=svm.SVC(kernel='poly',C=0.001,degree=2,gamma=0.05,decision_function_shape='ovr')
#训练多项式内核
print("start training pl")
model_poly.fit(data, label) # 训练
accuracy=model_poly.score(test_data,test_label)
print(accuracy)
joblib.dump(model_poly, 'pl.model')
print("save pl success")

model_rbf=svm.SVC(kernel='rbf',C=0.001,decision_function_shape='ovr')
#训练多项式内核
print("start training rbf")
model_rbf.fit(data, label) # 训练
accuracy=model_rbf.score(test_data,test_label)
print(accuracy)
joblib.dump(model_rbf, 'rbf.model')
print("save rbf success")
"""
