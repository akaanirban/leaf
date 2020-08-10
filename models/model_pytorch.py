#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 7/12/20 3:15 PM 2020

@author: Anirban Das
"""

"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import torch
from baseline_constants import ACCURACY_KEY
import copy
from utils.model_utils import batch_data
# from utils.tf_utils import graph_size
from thop import profile


class Model(ABC):

    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        self.seed = seed
        self._optimizer = optimizer
        self.net, self.criterion, self._optimizer = self.create_model(self.lr, momentum=0)
        torch.manual_seed(123 + self.seed)
        np.random.seed(self.seed)
        # https://github.com/sovrasov/flops-counter.pytorch/issues/16
        macs, params = profile(self.net, inputs=torch.rand(1, 1, 1, 28, 28))
        self.flops = macs * 2

    # def set_params(self, model_params):
    #     with self.graph.as_default():
    #         all_vars = tf.trainable_variables()
    #         for variable, value in zip(all_vars, model_params):
    #             variable.load(value, self.sess)

    # def copyParams(module_src, module_dest):
    #     params_src = module_src.named_parameters()
    #     params_dest = module_dest.named_parameters()
    #
    #     dict_dest = dict(params_dest)
    #
    #     for name, param in params_src:
    #         if name in dict_dest:
    #             dict_dest[name].data.copy_(param.data)

    def set_params(self, model):
        self.net = copy.deepcopy(model)
        self._optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0)

    def get_params(self):
        return copy.deepcopy(self.net)

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0)
        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 3-tuple consisting of:
                net: The neural network instantiated
                criterion: The training loss criterion
                optimizer: The training optimizer
        """
        return None, None, None

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of tensors which are parameters of the network or the weights
        """
        total_loss = 0
        for _ in range(num_epochs):
            loss = self.run_epoch(data, batch_size)
            total_loss += loss
            # print(f"Client {self} : Avg loss: {loss}, {total_loss/num_epochs}")
        update = self.get_params().state_dict()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return comp, update

    def run_epoch(self, data, batch_size):
        running_loss = 0
        count = 1
        self.net.train()
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            self.net.to("cuda:0")
            input_data = self.process_x(batched_x).to("cuda:0")
            target_data = self.process_y(batched_y).to("cuda:0")
            self._optimizer.zero_grad()
            self.outputs = self.net(input_data)
            loss = self.criterion(self.outputs, target_data)
            loss.backward()
            self._optimizer.step()
            running_loss += loss
            count += 1
        self.net.to("cpu")  # just to save gpu memory
        return running_loss / count

    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps)

    def cross_entropy(self, X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        p = self.softmax(X)
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def test(self, data, batch_size=50):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        """Well, in my case, used with torch.no_grad(): (train model), output.to("cpu") and torch.cuda.empty_cache() and this problem solved."""
        total_loss = 0
        acc = 0
        counter = 1
        samples = 0
        self.net.eval()
        with torch.no_grad():
            self.net.cuda()
            x_vecs = self.process_x(data['x']).to("cuda:0")
            labels = self.process_y(data['y']).to("cuda:0")
            outputs = self.net(x_vecs)
            predicted = torch.argmax(outputs, axis=1)
            loss = self.criterion(outputs, labels).to("cpu")
            correct = predicted.eq(labels).sum().to("cpu")
            acc += correct.item()
            total_loss += loss.item()
            samples = labels.shape[0]

        return {ACCURACY_KEY: acc / samples, 'loss': total_loss / counter}

    def close(self):
        self.sess.close()

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return torch.tensor(raw_x_batch)

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return torch.tensor(raw_y_batch)


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        with self.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                var_vals[v.name] = val
        for c in clients:
            with c.model.graph.as_default():
                all_vars = tf.trainable_variables()
                for v in all_vars:
                    v.load(var_vals[v.name], c.model.sess)

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()
