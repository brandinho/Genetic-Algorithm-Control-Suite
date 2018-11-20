import numpy as np
import tensorflow as tf

def shape_parameters(flattened_parameters, parameter_shapes, conv_net_dim):
    parameters_reshaped = []
    current_idx = 0
    for ps in range(len(parameter_shapes)):
        parameters_reshaped.append(flattened_parameters[current_idx:current_idx+parameter_shapes[ps]].reshape(conv_net_dim[ps]))
        current_idx += parameter_shapes[ps]
    return parameters_reshaped

class Actor():
    def __init__(self, sess, input_dimension, output_dimension, population_size):
        self.sess = sess
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        
        self.neurons = [36,24]
        self.std1 = np.sqrt(2/(self.input_dimension + self.neurons[0]))
        self.std2 = np.sqrt(2/(self.neurons[0] + self.neurons[1]))
        self.std3 = np.sqrt(2/(self.neurons[1] + self.output_dimension))   
        self.scaling_factor = np.hstack((np.array([self.std1]*(self.input_dimension * self.neurons[0])), np.array([1] * self.neurons[0]),
                                         np.array([self.std2]*(self.neurons[0] * self.neurons[1])), np.array([1] * self.neurons[1]),
                                         np.array([self.std3]*(self.neurons[1] * self.output_dimension)), np.array([1] * self.output_dimension)))

        self.population = np.random.multivariate_normal(mean = [0]*self.scaling_factor.shape[0], 
                                                        cov = np.diag(self.scaling_factor),
                                                        size = population_size)       
        
        self.inputs = tf.placeholder(shape = [None, self.input_dimension], dtype = tf.float32, name = "Inputs")

        self.weights1 = tf.Variable(tf.random_normal([self.input_dimension, self.neurons[0]]), name = "Weights1")
        self.bias1 = tf.Variable(tf.zeros([1,self.neurons[0]]) + 0.01, name = "Bias1")
        self.layer1 = tf.nn.relu(tf.matmul(self.inputs, self.weights1) + self.bias1, name = "Layer1")
        
        self.weights2 = tf.Variable(tf.random_normal([self.neurons[0], self.neurons[1]]), name = "Weights2")
        self.bias2 = tf.Variable(tf.zeros([1,self.neurons[1]]) + 0.01, name = "Bias2")
        self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.weights2) + self.bias2, name = "Layer2")        

        self.weights3 = tf.Variable(tf.random_normal([self.neurons[1], self.output_dimension]), name = "Weights3")
        self.bias3 = tf.Variable(tf.zeros([1,self.output_dimension]) + 0.01, name = "Bias3")
        self.policy = tf.nn.tanh(tf.matmul(self.layer2, self.weights3) + self.bias3, name = "Policy")
        
        self.update_placeholders = [tf.placeholder(shape = s.shape, dtype = tf.float32) for s in tf.trainable_variables()]
        self.update_variables = [tf.trainable_variables()[i].assign(self.update_placeholders[i]) for i in range(len(self.update_placeholders))]        

    def take_action(self, inputs):
        return self.sess.run(self.policy, {self.inputs: inputs})
    
    def update_parameters(self, new_parameters, parameter_shapes, neural_net_dim):
        model_parameters = shape_parameters(new_parameters, parameter_shapes, neural_net_dim)
        self.sess.run(self.update_variables, {i: d for i, d in zip(self.update_placeholders, model_parameters)})
