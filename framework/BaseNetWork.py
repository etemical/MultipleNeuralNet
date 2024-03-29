import numpy as np
from inspect import isfunction

class Module:
    # 总层数，包括网络层和激活函数都算在内
    __layers = []  # 缓存每层的权重和偏置参数以及激活函数
    __cache_input = []  #缓存每层的输入数据

    def __call__(self, x):
        return self.forward(x)

    def forward(self , x):
        return x

    @classmethod
    def clear(cls):
        Module.__cache_input.clear()
        Module.__layers.clear()

    @classmethod
    def layer_count(cls):
        return len(cls.__layers)

    @classmethod
    def addLayer(cls, layer):
        Module.__layers.append(layer)

    @classmethod
    def addCache_input(cls, input_x):
        Module.__cache_input.append(input_x)

    def cache_data(self):
        """
        默认返回从倒数第一层开始
        返回每一层的输入数据
        相当于是上一层的输出数据
        :return:
        """
        for data in reversed(self.__cache_input):
            yield data

    def parameters(self):

        """
        获取每个层对象 默认反向获取保存的网络参数，即从后往前遍历
        通过yield 用next方法一次返回一个层对象
        :return:
        """
        for e in reversed(self.__layers):
            yield e  # yield 语法类似于return ，通过next一次返回一个params，可以减少一次性生成数据所带来的内存开销


class Functional:

    # 带参的装饰器
    # 把激活函数记录到计算图里面去
    def OperGraph(active):

        # 装饰器只要加载了就会立即执行，参数是被包裹的函数
        def decorate(fn):
            # 包裹被修饰函数的增强方法，此函数在被包裹函数调用时执行，参数是被包裹函数的参数
            def advice(*args):
                Module.addLayer(active) #  把计算操作步骤项添加到计算图里
                Module.addCache_input(args[1]) # 把激活函数的输入也加入到缓存输入里
                return fn(*args)
            return advice
        return decorate

    # 不带参的装饰器
    def OperGraph2(fn):

        # 装饰器只要加载了就会立即执行吃，参数是被包裹的函数
        # 包裹被修饰函数的增强方法，此函数在被包裹函数调用时执行，参数是被包裹函数的参数
        def advice(*args):
            Module.addLayer(fn) #  把计算操作步骤项添加到计算图里
            return fn(*args)
        return advice

    @classmethod
    @OperGraph("sigmoid")
    def sigmoid(cls, z):
        s = 1 / (1 + np.exp(-z))
        return s

    @classmethod
    @OperGraph("relu")
    def relu(cls, z):
        return np.maximum(0, z)

    @classmethod
    @OperGraph("tanh")
    def tanh(cls, z):
        s = np.divide(np.exp(z) - np.exp(-z), np.exp(z) + np.exp(-z))
        return s

    @classmethod
    @OperGraph("softmax")
    def softmax(cls, z):
        """
        找出z中每列对应的最大值
        然后各个列减去对应维度的最大值以达到缩小softmax指数运算的结果
        太大计算机会显示NaN
        :param z:
        :return:
        """
        max = np.max(z, axis=0, keepdims=True)
        a = np.divide(np.exp(z-max) , np.sum(np.exp(z-max), axis=0))
        return a

    @classmethod
    def sigmoid_grad(cls, a):
        return a * (1 - a)

    @classmethod
    def tanh_grad(cls, a):
        return 1 - np.power(a, 2)

    @classmethod
    def relu_grad(cls, a):
        t = np.copy(a)
        t[t > 0] = 1
        return t

    @classmethod
    def softmax_grad(cls, a, Y):
        """
        j为输出，i为输入
        DiSj 表示为第j个输出对i个输入的导数
        如果j==i， DiSj=Aj*(1-Aj)
        如果j!=i,  DiSj=-Aj*Ai
        要分段求，比较繁琐
        :param a: softmax的输出，一组向量
        :param Y: 标签 一组向量（一般而言由1和0组成的）
        :return: softmax的梯度
        """

        j = np.argmax(Y, axis=0)
        grads = np.zeros_like(a).T
        a = a.T
        #  找出每个样本最大值对应的输出
        Aj = a[np.arange(0, a.shape[0]), j]
        # 当j不等于i的时候
        grads = -a * Aj.reshape(-1,1)
        # 当j等于i的时候
        grads[np.arange(0, grads.shape[0]) , j] = Aj * (1. - Aj)
        return grads.T


    @classmethod
    def getNode_grad(cls, node, *next_layer_input, Y):
        if isinstance(node ,str):
            if node == "sigmoid":
                return cls.sigmoid_grad(*next_layer_input)
            elif node == "relu":
                return cls.relu_grad(*next_layer_input)
            elif node == "tanh":
                return cls.tanh_grad(*next_layer_input)
            elif node == "softmax":
                return cls.softmax_grad(*next_layer_input, Y)

        if isinstance(node, Module):
            return 1


if __name__ == '__main__':

    a = Functional.sigmoid(-6)
    print(a)
    print(isfunction(Module._Module__layers[0]))
    print(hasattr(Module._Module__layers[0],"__call__"))
    print(Module._Module__layers[0], Module._Module__cache_input[0])
    z = np.arange(10)

    a = Functional.softmax(z)
    a = a.round(6)
    print(a.round(6))
    print(np.argmax(a))
    y = np.array([0,0,0,0,0,0,0,0,0,1])




    z2 = np.random.rand(10)
    z2 = np.array([0.12410648, 0.11217966, 0.0664022,  0.12460169, 0.11754158, 0.0776272,0.05509569, 0.10349527,0.14481626, 0.07413397])


    a2 = Functional.softmax(z2)

    a3 = np.zeros((2,10))
    a3[0] = a
    a3[1] = a2

    print(np.argmax(a2))
    y2 = np.array([0,0,1,0,0,0,0,0,1,0])
    y3 = np.zeros((2,10))
    y3[0] = y
    y3[1] = y2
    print(a3)
    print(y3)

    # 2.5266892886538392
    loss = -np.sum(y3*np.log(a3)) / 2
    print(loss)


    a = np.array([[0.2, 0.1, 0.9], [0.3, 0.4, 0.6]], dtype=np.float32)
    y = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32)
    loss = -np.sum(y*np.log(a)) / 2
    print(loss)
    loss = np.sum(-np.sum(np.log(a), axis=1) / 3) / 2

    print(loss)
    print(-np.sum(np.log(a), axis=1))
    print(sum([4.0173836/3. ,2.6310892/3.]) / 2)
    pass