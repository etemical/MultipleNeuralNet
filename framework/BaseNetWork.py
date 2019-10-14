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
    def getNode_grad(cls, node, *next_layer_input):
        if isinstance(node ,str):
            if node == "sigmoid":
                return cls.sigmoid_grad(*next_layer_input)
            elif node == "relu":
                return cls.relu_grad(*next_layer_input)
            elif node == "tanh":
                return cls.tanh_grad(*next_layer_input)
        if isinstance(node, Module):
            return 1



if __name__ == '__main__':

    a = Functional.sigmoid(-6)
    print(a)
    print(isfunction(Module._Module__layers[0]))
    print(hasattr(Module._Module__layers[0],"__call__"))
    print(Module._Module__layers[0], Module._Module__cache_input[0])

    pass