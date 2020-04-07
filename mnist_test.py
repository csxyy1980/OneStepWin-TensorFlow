#-*- coding:utf8 -*-
# noinspection PyUnresolvedReferences
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


import matplotlib.pyplot as plt
image_index = 1234
plt.imshow(x_train[image_index],cmap='Greys')
plt.show()
print(y_train[image_index])

import numpy as np
x_train = np.pad(x_train,((0,0),(2,2),(2,2)),'constant',constant_values=0)
print(x_train.shape)
x_train = x_train.astype('float32')
x_train /=255
x_train=x_train.reshape(x_train.shape[0],32,32,1)
print(x_train.shape)

x_test = np.pad(x_test,((0,0),(2,2),(2,2)),'constant',constant_values=0)
print(x_test.shape)
x_test = x_test.astype('float32')
x_test /=255
x_test=x_test.reshape(x_test.shape[0],32,32,1)
print(x_test.shape)


# LeNet模型构建
# 定义模型
# 模型的构建:  tf.keras.Model 和tf.keras.layers
# 模型的损失函数：tf.keras.losses
# 模型的优化器：   tf.keras.optimizer
# 模型的评估：   tf.keras.metrics
class LetNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #此处添加初始化代码（包含call方法中会用到的层）
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5,5),
            padding='valid',
            activation=tf.nn.relu)
        self.pool_layer_1=tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')

        self.conv_layer_2=tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5,5),
            padding='valid',
            activation=tf.nn.relu)
        self.pool_layer_2=tf.keras.layers.MaxPooling2D(padding='same')
        self.flatten=tf.keras.layers.Flatten()
        self.fc_layer_1=tf.keras.layers.Dense(
            units=120,
            activation=tf.nn.relu)
        self.fc_layer_2=tf.keras.layers.Dense(
            units=84,
            activation=tf.nn.relu)
        self.output_layer=tf.keras.layers.Dense(
            units=10,
            activation=tf.nn.softmax)
    def call(self,inputs):    #[batch_size,28,28,1]
        x=self.conv_layer_1(inputs)
        x=self.pool_layer_1(x)
        x=self.conv_layer_2(x)
        x=self.pool_layer_2(x)
        x=self.flatten(x)
        x=self.fc_layer_1(x)
        x=self.fc_layer_2(x)
        output=self.output_layer(x)

        return output
    #还可以添加自定义的方法
#模型实例化方法
# model=tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),padding='valid',activation=tf.nn.relu,input_shape=(32,32,1)),
#     tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same'),
#     tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation=tf.nn.relu),
#     tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=120,activation=tf.nn.relu),
#     # tf.keras.layers.Conv2D(filters=120,kernel_size=(5,5),strides=(1,1),activation='tanh',padding='valid'),
#     # tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=84,activation=tf.nn.relu),
#     tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
# ])
model=LetNet()

#模型展示
model.summary()

#模型训练
#import numpy as np

#超参数配置
num_epochs=10
batch_size=64
learning_rate=0.001

#优化器
adam_optimizer=tf.keras.optimizers.Adam(learning_rate)

model.compile(optimizer=adam_optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

import datetime
start_time=datetime.datetime.now()

model.fit(x=x_train,
          y=y_train,
          batch_size=batch_size,
          epochs=num_epochs)
end_time=datetime.datetime.now()
time_cost=end_time-start_time
print("time_cost = ",time_cost) #CPU time cost: 5min, GPU time cost: less than 1min

#模型保存
model.save('C:\\Users\\csxyy\\Documents\\lenet_model.h5')

#评估指标
print(model.evaluate(x_test, y_test))  #loss value & metrics value
#预测
image_index=4444
print(x_test[image_index].shape)
plt.imshow(x_test[image_index].reshape(32,32),cmap='Greys')

pred = model.predict((x_test[image_index].reshape(1,32,32,1)))
print(pred.argmax())


# ##############################################
# 以下为自己写的手写数字拍照处理识别的代码

#自测手写
import cv2
#第一步 读取图片
img=cv2.imread('/Users/weiqi/Documents/TensorflowLearning/tensorflow2.0demo/8.jpg')
print(img.shape)
#第二步 将图片转为灰度图
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img.shape)
plt.imshow(img,cmap='Greys')
plt.show()
#第三步 将图片的底色和字的颜色取反
img=cv2.bitwise_not(img)
plt.imshow(img,cmap='Greys')
plt.show()
#第四部 将底变成纯白色，将字变成纯黑色
img[img<140]=0
img[img>140]=255
plt.imshow(img,cmap='Greys')
plt.show()
#第五步 将图片尺寸缩放为输入规定尺寸
img=cv2.resize(img,(32,32))
#第六步 将数据类型转为float32
img=img.astype('float32')
#第七步 数据正则化
img /=255
#第八步 增加维度为输入的规定格式
img = img.reshape(1,32,32,1)
print(img.shape)
#第九步 预测
pred=model.predict(img)
#第十步 输出结果
print(pred.argmax())
