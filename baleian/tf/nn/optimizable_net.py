import baleian.tf.nn.neural_net as nn

from abc import *


class OptimizableNetwork(nn.NeuralNetwork):

    def __init__(self, input_size):
        super(OptimizableNetwork, self).__init__(input_size)
        self.optimizer = None
        self.learning_rate = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self


class AbstractModelSession(nn.AbstractModelSession, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(AbstractModelSession, self).__init__(*args, **kwargs)
        self.optimizer = None
        self.learning_rate = None
        self.cost = None
        self.optimize = None

    def init_layer(self, net: OptimizableNetwork):
        super(AbstractModelSession, self).init_layer(net)
        self.optimizer = net.optimizer
        self.learning_rate = net.learning_rate

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
        self._init_optimize()

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate
        self._init_optimize()

    @property
    def cost(self):
        return self.__cost

    @cost.setter
    def cost(self, cost):
        self.__cost = cost
        self._init_optimize()

    def _init_optimize(self):
        if self.optimizer is None:
            return
        if self.learning_rate is None:
            return
        if self.cost is None:
            return
        self.optimize = self.optimizer(self.learning_rate).minimize(self.cost)

    @abstractmethod
    def training(self, *args, **kwargs):
        """Overridable functional interface for training this model

        :param args: to make feed_dict properties
        :param kwargs: other optional arguments
        :return:
        """
        pass

    @abstractmethod
    def get_cost(self, *args, **kwargs):
        pass
