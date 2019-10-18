import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time
from framework.BaseNetWork import Module
from framework.BaseNetWork import Functional as F

class Linear(Module):

    def __init__(self,in_feature, out_feature, bias=True):

        #  初始化当前层的权重矩阵
        self.weight = np.random.randn(out_feature, in_feature)  / np.sqrt(in_feature)
        self.bias = np.random.randn(out_feature, 1) * 0.01

    def __call__(self, x):

        return self.forward(x)

    def forward(self,  x):
        """
        x：input shape must be in_feature , Numbers structure (V, N)
        V is H*W*C
        :param x:
        :return:
        """
        # 把当前层和输入的数据添加到计算图里面
        Module.addCache_input(x)
        Module.addLayer(self)
        z = np.dot(self.weight, x) + self.bias
        return z

class MultiNetWork(Module):

    def __init__(self):

        self.layer1 = Linear(64*64*3, 16)
        self.layer2 = Linear(16, 8)
        self.layer3 = Linear(8, 2)

    def forward(self , x):
        Z1 = self.layer1(x)
        # 中间层用relu激活函数来激活一下
        A1 = F.relu(Z1)
        Z2 = self.layer2(A1)
        A2 = F.relu(Z2)
        Z3 = self.layer3(A2)
        A3 = F.softmax(Z3)

        # # 最后一层用sigmoid当输出函数输出概率
        # A3 = F.sigmoid(Z3)
        return A3

class Optimizer:

    def __init__(self, net, lr=0.01):
        self.net = net
        self.lr = lr

    def step(self, grads):
        # 从倒数第一层开始 顺序执行更新参数
        params = self.net.parameters()
        # 获取总层数
        index = Module.layer_count()
        try:
            while True:
                layer = next(params)
                # 判断是否属于网络层，如果是则更新梯度
                if isinstance(layer, Module):
                    layer.weight = layer.weight - self.lr * grads["dW"+str(index)]
                    layer.bias = layer.bias - self.lr * grads["db"+str(index)]
                index -= 1
        except Exception as e:
            pass
        finally:
            Module.clear() # 清空网络所有数据
            grads.clear() # 清空梯度


class BCELoss(Module):
    """
    二分类交叉熵损失函数
    配合Sigmoid来用，用在输出层
    输出一个二分类的概率
    """
    def __call__(self, output, target):
        assert(output.ndim == 2)
        self.m = output.shape[1] # 样本数
        return self.__loss(output, target, self.m)

    def __loss(self, A, Y, m):
        """
        求所有样本的平均损失
        :param A: 输出
        :param Y: 标签
        :param m: 样本数
        :return: 平均损失
        """
        self.A = A + (1e-5)
        self.Y = Y
        self.loss_value = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return self

    def float(self):
        return self.loss_value

    def __str__(self):
        if hasattr(self,"loss_value"):
            return self.loss_value
        return super().__str__()

    def backward(self):
        """
        多层网络反向传播的核心代码
        这里可以是任意数量的网络层，从最后一层开始反向求导
        从后往前一层层计算梯度，采用链式法则
<<<<<<< HEAD
        :return: grads 字典（每层的梯度）
=======
        :return: grads 字典
>>>>>>> 85f00821fe124de1a29900784020c97b437fffae
        """

        nodes = self.parameters()
        data = self.cache_data()

        #  包含了激活层在内的所有层
        current_layer = layer_count = Module.layer_count()
        grads = {} # 保存网络所有层的梯度
        current_input = {}

        try:

            temp = []  # 存储网络的层及其激活函数
            # 每次遍历一小部分（该部分是一层网络层和它的激活函数（如果有的话））
            while True:
                #  倒数第一层
                if current_layer == layer_count:
                    dA = (1 - self.Y) / (1 - self.A) - (self.Y / self.A)
                    current_input["X" + str(current_layer + 1)] = self.A

                node = next(nodes)
                # 获取当前层的输入
                x = next(data)
                # 获取当前节点的梯度
                dA_dZ_grad = F.getNode_grad(node, current_input["X" + str(current_layer + 1)])
                dZ = dA * dA_dZ_grad
                dA = dZ
                if isinstance(node, Module):
                    w = node.weight  # w
                    dW = np.dot(dZ, x.T) / self.m
                    db = np.sum(dZ, axis=1, keepdims=True) / self.m
                    grads["dW" + str(current_layer)] = dW
                    grads["db" + str(current_layer)] = db
                    dA = np.dot(w.T, dZ)

                current_input["X" + str(current_layer)] = x
                current_layer -= 1
                del node, dA_dZ_grad, dZ, x
            del dA
            # return grads
        except:
            pass
        finally:
            return grads
class CrossEntropyLoss(Module):

    """
    多分类交叉熵损失函数
    配合Softmax来用，用在输出层
    输出一个多分类的概率
    """
    def __call__(self, output, target):
        """
        target必须要转成one-hot形式
        :param output:
        :param target:
        :return:
        """

        assert (output.ndim == 2)
        self.m = output.shape[1]  # 样本数
        return self.__loss(output, target, self.m)

    def __loss(self, A, Y, m):
        """
        求所有样本的平均损失
        用交叉熵公式 L=-ΣYlogA
        由于多分类标签one-hot一个为1，其余为0，所以求和就变成了L=-YlogA(一个样本的损失)
        而批样本的平均损失就是L=-1/m*ΣYlogA
        :param A: 输出
        :param Y: 标签
        :param m: 样本数
        :return: 平均损失
        """
        self.A = A
        self.Y = Y
        self.loss_value = -np.mean(np.sum(self.Y * np.log(self.A), axis=0))
        # print(-np.sum(self.Y * np.log(self.A)) / m)
        return self

    def float(self):
        return self.loss_value

    def __str__(self):
        if hasattr(self, "loss_value"):
            return self.loss_value
        return super().__str__()

    def backward(self):
        """
        多层网络反向传播的核心代码
        这里可以是任意数量的网络层，从最后一层开始反向求导
        从后往前一层层计算梯度，采用链式法则
        :return: grads 字典（每层的梯度）
        """
        nodes = self.parameters()
        data = self.cache_data()

        #  包含了激活层在内的所有层
        current_layer = layer_count = Module.layer_count()
        grads = {}  # 保存网络所有层的梯度
        current_input = {}

        try:
            temp = []  # 存储网络的层及其激活函数
            # 每次遍历一小部分（该部分是一层网络层和它的激活函数（如果有的话））
            while True:
                #  倒数第一层
                if current_layer == layer_count:
                    y = self.Y.T
                    a = self.A.T
                    j = np.argmax(y, axis=1) # 找出y中最大值对应的下标
                    # L对A求导 就是 -Yj/Aj
                    dA = -y[np.arange(0, self.m), j] / a[np.arange(0, self.m), j]
                    current_input["X" + str(current_layer + 1)] = self.A

                # 反向迭代计算图中的节点
                node = next(nodes)
                # 获取当前层的输入
                x = next(data)
                # 获取当前节点的梯度
                dA_dZ_grad = F.getNode_grad(node, current_input["X" + str(current_layer + 1)], Y=self.Y)
                dZ = dA * dA_dZ_grad
                dA = dZ

                if isinstance(node, Module):
                    w = node.weight  # w
                    dW = np.dot(dZ, x.T) / self.m
                    db = np.sum(dZ, axis=1, keepdims=True) / self.m
                    grads["dW" + str(current_layer)] = dW
                    grads["db" + str(current_layer)] = db
                    dA = np.dot(w.T, dZ)

                current_input["X" + str(current_layer)] = x
                current_layer -= 1
                del node, dZ, dA_dZ_grad, x
            del dA
        except Exception as e:
            pass
        finally:
            return grads
class Train:

    def __init__(self):
        # np.random.seed(1)
        self.net = MultiNetWork()
        self.loss_func = CrossEntropyLoss()
        self.optimizer = Optimizer(self.net, lr=0.01)

    def get_train_dataset(self):
        train_set = h5py.File("../datasets/train_catvnoncat.h5", "r")
        train_data = train_set["train_set_x"][:] / 255.
        train_data = self.normalization(train_data)
        train_target = train_set["train_set_y"][:]
        return train_data, train_target

    def get_test_dataset(self):

        test_set = h5py.File("../datasets/test_catvnoncat.h5", "r")
        test_data = test_set["test_set_x"][:] / 255.
        test_data = self.normalization(test_data)
        test_target = test_set["test_set_y"][:]
        return test_data, test_target

    def normalization(self, data):
        mean = [0.4413, 0.4244, 0.3560]
        std = [0.26870, 0.2512, 0.2685]
        mean = np.array(mean)
        std = np.array(std)
        mean = mean.reshape((1,1,1,3))
        std = std.reshape((1,1,1,3))
        return (data - mean) / std

    def deNormalization(self, data):
        mean = [0.4413, 0.4244, 0.3560]
        std = [0.26870, 0.2512, 0.2685]
        mean = np.array(mean)
        std = np.array(std)
        return data * std + mean

    """
    第0次优化，损失是:0.698277
    第100次优化，损失是:0.486761
    第200次优化，损失是:0.233869
    第300次优化，损失是:0.096318
    第400次优化，损失是:0.052083
    """

    def to_one_hot(self, C,N, target):
        """
        target
        :param C: 为输出的类别个数
        :param N: 为样本数
        :param target:
        :return:
        """
        one_hot = np.eye(C,N).T
        # one_hot后面的下标表示指定每行的1在哪个位置处
        one_hot = one_hot[target]
        return one_hot.T

    def train(self):
        input , target = self.get_train_dataset()
        input = input.reshape(input.shape[0], -1).T  # 把输入形状变换为 N V结构再转置成V，N结构
        # target = target.reshape(1, -1) # 用的sigmoid为输出函数，损失函数为BCELoss
        target = self.to_one_hot(2,target.size, target) # 转one-hot
        cost = []
        for i in range(5000):
            output = self.net(input)
            loss = self.loss_func(output, target)
            grads = loss.backward()
            self.optimizer.step(grads)
            if i % 100 == 0:
                print("第{}次优化，损失是:{}".format(i, loss.float()))
                cost.append(loss.float())
                plt.clf()
                plt.plot(cost)
                plt.pause(0.1)
        self.save(self.net, "../models/net_by_softmax2.bin")

    def save(self, net, path):
        with open(path, "wb") as f:
            pickle.dump(net, f)

    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_softmax(self):

        """
        拿测试集做预测
        :return: 输出预测的正确率
        """
        input, target = self.get_test_dataset()
        net = self.load("../models/net_by_softmax2.bin")
        # # 传入训练好的参数，实例化网络
        x = input
        input = input.reshape(input.shape[0], -1).T # reshape to N V  再转置成V，N结构
        target = np.expand_dims(target, axis=0)
        output = net(input)
        # 把结果拿来做预测，1就是有猫，0就是没猫
        # print(output.round())
        # print(target)
        # prediction = np.where(output >= 0.5, 1, 0)

        prediction = np.argmax(output,axis=0)

        """ 要做图片还原 均值和归一化 """
        # x = self.deNormalization(x) * 255
        # x = x.astype(np.uint8)
        # for i in range(x.shape[0]):
        #     plt.clf()
        #     plt.axis("off")
        #     plt.imshow(x[i])
        #     plt.text(0, -2, "cat" if prediction[i] == 1 else "non-cat", fontsize=20, color="red")
        #     plt.pause(1)

        # print(prediction)
        # 当然用 output.round() 也可以，直接四舍五入了返回0，1的结果
        result = (prediction == target).mean()
        print("正确率:", str(result * 100) + "%")
        img = Image.open(r"D:/catdog_img/bg_pic/pic74.jpg")
        img = img.convert("RGB")
        img = img.resize((64, 64), Image.ANTIALIAS)
        img = np.array(img) / 255.
        img = img[np.newaxis, :]  # 等价于 np.expand_dims() 添加一个维度

        img = self.normalization(img)
        img = img.reshape(1, -1).T
        output = net(img)
        print(output)
        if np.argmax(output, axis=0) == 1:
            print("猫，置信度为:{}".format(str(np.max(output).item() * 100) + "%"))
        else:
            print("不是猫,置信度为:{}".format(str(np.max(output).item() * 100) + "%"))


    def predict(self):
        """
        拿测试集做预测
        :return: 输出预测的正确率
        """
        input, target = self.get_test_dataset()
        net = self.load("../models/net_4.bin")
        # # 传入训练好的参数，实例化网络
        x = input
        input = input.reshape(input.shape[0], -1).T # reshape to N V  再转置成V，N结构
        target = np.expand_dims(target, axis=0)
        output = net(input)
        # 把结果拿来做预测，1就是有猫，0就是没猫
        # print(output.round())
        # print(target)
        # prediction = np.where(output >= 0.5, 1, 0)
        prediction = output.round()
        """ 要做图片还原 均值和归一化 """
        # x = self.deNormalization(x) * 255
        # x = x.astype(np.uint8)
        # for i in range(x.shape[0]):
        #     plt.clf()
        #     plt.axis("off")
        #     plt.imshow(x[i])
        #     plt.text(0, -2, "cat" if prediction[0, i] > 0 else "non-cat", fontsize=20, color="red")
        #     plt.pause(1)

        # print(prediction)
        # 当然用 output.round() 也可以，直接四舍五入了返回0，1的结果
        result = (prediction == target).mean()
        print("正确率:", str(result * 100) + "%")
        img = Image.open(r"D:/catdog_img/bg_pic/pic14.jpg")
        img = img.convert("RGB")
        img = img.resize((64, 64), Image.ANTIALIAS)
        img = np.array(img) / 255.
        img = img[np.newaxis, :]  # 等价于 np.expand_dims() 添加一个维度
        img = self.normalization(img)
        img = img.reshape(1, -1).T
        output = net(img)
        if output.round().item() == 1:
            print("猫，置信度为:{}".format(str(output.item() * 100) + "%"))
        else:
            print("不是猫,置信度为:{}".format(str(output.item() * 100) + "%"))

"""
    第0次优化，损失是:0.698277
    第100次优化，损失是:0.486761
    第200次优化，损失是:0.233869
    第300次优化，损失是:0.096318
    第400次优化，损失是:0.052083
"""

if __name__ == '__main__':
    # np.random.seed(1)
    t = Train()
    t.train()
    # print(Module._Module__layers)
    # time.sleep(30)
    t.predict_softmax()
    # t.predict()
    # print(type(t.loss_func).__name__)
    # print(t.get_test_dataset())
    # param = obj.parameters()
    # p = next(param)
    # print(p.weight , p.bias)
    # p2 = next(param)
    # print(p2.weight, p2.bias)
    # p2.weight = p2.weight * 2
    # print(obj.layer2.weight)
    #
    # param = obj.parameters()
    #
    # print(list(param)[1].weight, )
    # x = np.linspace(-1, 1 , 10).reshape(10,1)
    # output = obj(x)
    # print(output)
    # print(type(t.loss_func))


