# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 23:55:04 2022

@author: QV-137
"""

import numpy as np

# Neurona para saber si un usuario tiene mayor frecuencia cardiaca por alguna actividad física

# Primero creamos nuestra clase TLU
class TLU():
    def __init__(self, inputs, weights):
        """Class constructor.
        
        Parameters
        ----------
        inputs : list
            List of input values.
        weights : list
            List of weight values.
        """

        self.inputs = np.array(inputs) # TODO: np.array <- inputs ::: Velocidad, respiración, ritmo cardiaco, etc
        self.weights = np.array(weights) # TODO: np.array <- weights ::: Importancia o peso de cada variable
  
    def decide(self, treshold):
        """Function that operates inputs @ weights.
        
        Parameters
        ----------
        treshold : int
            Threshold value for decision.
        """

        # TODO: Inner product of data
        n = len(self.inputs)
        result = 0
        for index in range(n):
            result += self.inputs[index] * self.weights[index]
        # result = self.inputs[index] @ self.weights[index] ::: De esta manera se puede reescribir el producto punto de ambos vectores sin usar ciclos for
        if result >= threshold:
            return 1
        else:
            return 0
        # return int(result >= threshold) ::: De esta manera se puede reescribir la condicional para conocer si el resultado es verdadero o falso

# Now, we need to set inputs and weights
inputs, weights = [], []

questions = [
    "· ¿Cuál es la velocidad? ",
    "· ¿Ritmo cardiaco? ",
    "· ¿Respiración? "
]

for question in questions:
    i = int(input(question))
    w = int(input("· Y su peso asociado es... "))
    inputs.append(i)
    weights.append(w)
    print()

threshold = int(input("· Y nuestro umbral/límite será: "))

artificial_neuron = TLU(inputs, weights) # TODO Instantiate Perceptron
artificial_neuron.decide(threshold) # TODO Apply decision function with threshold

artificial_neuron.inputs

# Modificamos para añadir la función de activación
class Perceptron():
    def __init__(self, inputs, weights):
        """Class constructor.
        
        Parameters
        ----------
        inputs : list
            List of input values.
        weights : list
            List of weight values.
        """

        self.inputs = np.array(inputs) # TODO: np.array <- inputs
        self.weights = np.array(weights) # TODO: np.array <- weights
  
    def decide(self, bias):
        """Function that operates inputs @ weights.
        
        Parameters
        ----------
        bias : int
            The bias value for operation.
        """

        # TODO: Inner product of data + bias
        z = self.inputs @ self.weights + bias
        # TODO: Apply sigmoid function f(z) = 1 / (1 + e^(-z))
        return 1/ (1 + np.exp(-z))

bias = int(input("· El nuevo bias será: "))
perceptron = Perceptron(inputs, weights)
perceptron.decide(bias)

class TrainableNeuron():
    def __init__(self, n):
        """Class constructor.
        
        Parameters
        ----------
        n : int
            Input size.
        """
        
        np.random.seed(123)
        self.synaptic_weights = 2 * np.random.random((n, 1)) - 1 # TODO. Use np.random.random((n, 1)) to gen values in (-1, 1)
        # 2* ... - 1 Puesto que al mutlipicar por 2 el int, es [0,2] y al restar -1 es [-1,1]

    def __sigmoid(self, x):
        """Sigmoid function.
        
        Parameters
        ----------
        x : float
            Input value to sigmoid function.
        """
        
        # TODO: Return result of sigmoid function f(z) = 1 / (1 + e^(-z))
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        """Derivative of the Sigmoid function.
        
        Parameters
        ----------
        x : float
            Input value to evaluated sigmoid function."""

        # TODO: Return the derivate of sigmoid function x * (1 - x)
        return x * (1 - x)

    def train(self, training_inputs, training_output, iterations):
        """Training function.
        
        Parameters
        ----------
        training_inputs : list
            List of features for training.
        training_outputs : list
            List of labels for training.
        iterations : int
            Number of iterations for training.
        
        Returns
        -------
        history : list
            A list containing the training history.
        """

        history = []
        
        for iteration in range(iterations):
            output = self.predict(training_inputs)
            error = training_output.reshape((len(training_inputs), 1)) - output
            #error = - training_output.reshape((len(training_inputs), 1)) * np.log(output) \
            #        - (1 - training_output.reshape((len(training_inputs), 1))) * output
            #error /= len(output)
            adjustment = np.dot(training_inputs.T, error *
                                self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

            history.append(np.linalg.norm(error))
        
        return history

    def predict(self, inputs):
        """Prediction function. Applies input function to inputs tensor.
        
        Parameters
        ----------
        inputs : list
            List of inputs to apply sigmoid function.
        """
        # TODO: Apply self.__sigmoid to np.dot of (inputs, self.synaptic_weights)
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

# Training samples:
input_values = [(0, 1), (1, 0), (0, 0)]   # TODO. Define the input values as a list of tuples
output_values = [1, 1, 0]  # TODO. Define the desired outputs

training_inputs = np.array(input_values)
training_output = np.array(output_values).T.reshape((3, 1))

# Initialize Sigmoid Neuron:
neuron = TrainableNeuron(2)
print("Initial random weights:")
neuron.synaptic_weights

# TODO.
# We can modify the number of epochs to see how it performs.
epochs = 10000

# We train the neuron a number of epochs:
history = neuron.train(training_inputs, training_output, epochs)
print("New synaptic weights after training: ")
neuron.synaptic_weights

import matplotlib.pyplot as plt
plt.style.use('seaborn')

x = np.arange(len(history))
y = history

plt.plot(x, y)

# We predict to verify the performance:
one_one = np.array((1, 1))
print("Prediction for (1, 1): ")
neuron.predict(one_one)

# Final
