import numpy as np

class Cognitron:
    def __init__(self, input_size, hidden_layer_sizes, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Инициализация весов
        self.W = {}
        self.b = {}
        self.dW = {}
        self.db = {}
        self.layers = []
        self.activation_func = lambda x: 1/(1+np.exp(-x))
        self.activation_func_prime = lambda x: x*(1-x)
        
        # Создание слоев
        self.create_layers()
    
    def create_layers(self):
        self.layers.append(np.random.uniform(-1, 1, (self.hidden_layer_sizes[0], self.input_size)))
        self.W['h0'] = self.layers[-1]
        self.b['h0'] = np.zeros(self.hidden_layer_sizes[0])
        self.dW['h0'] = np.zeros_like(self.W['h0'])
        self.db['h0'] = np.zeros_like(self.b['h0'])
        
        for i in range(1, len(self.hidden_layer_sizes)):
            self.layers.append(np.random.uniform(-1, 1, (self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])))
            self.W['hi'] = self.layers[-1]
            self.b['hi'] = np.zeros(self.hidden_layer_sizes[i])
            self.dW['hi'] = np.zeros_like(self.W['hi'])
            self.db['hi'] = np.zeros_like(self.b['hi'])
        
        self.layers.append(np.random.uniform(-1, 1, (self.output_size, self.hidden_layer_sizes[-1])))
        self.W['ho'] = self.layers[-1]
        self.b['ho'] = np.zeros(self.output_size)
        self.dW['ho'] = np.zeros_like(self.W['ho'])
        self.db['ho'] = np.zeros_like(self.b['ho'])
    
    def forward(self, inputs):
        a = {'i': inputs}
        z = {'i': inputs}
        for layer in range(len(self.hidden_layer_sizes)):
            W = self.W['hi' if 'h' in a else 'ho']
            b = self.b['hi' if 'h' in a else 'ho']
            a['h' if 'h' in a else 'o'] = self.activation_func(np.dot(W, a['i']) + b)
            z['h' if 'h' in a else 'o'] = np.dot(W, z['i']) + b
            a.pop('i')
            z.pop('i')
        return a['o'], z['o']
    
    def backward(self, targets):
        delta_L = self.activation_func_prime(self.z['o']) * (targets - self.a['o'])
        self.dW['ho'] += np.dot(delta_L, self.a['o'].reshape(-1, 1)).reshape(self.dW['ho'].shape)
        for layer in reversed(range(len(self.hidden_layer_sizes))):
            W = self.W['hi' if 'h' in z else 'ho']
            b = self.b['hi' if 'h' in z else 'ho']
            delta = np.dot(self.dW['hi' if 'h' in z else 'ho'].T, delta_L)
            delta_L = self.activation_func_prime(z['h' if 'h' in z else 'o']) * delta
            self.dW['hi' if 'h' in z else 'ho'] = np.dot(delta, z['h' if 'h' in z else 'o'].reshape(-1, 1)).reshape(self.dW['hi' if 'h' in z else 'ho'].shape)
            self.db['hi' if 'h' in z else 'ho'] += delta_L
            return self.dW, self.db

    def train(self, inputs, targets, iterations=1000):
        for i in range(iterations):
            if i % 100 == 0:
                print(f"Iteration: {i}")
            output, z = self.forward(inputs)
            self.backward(targets)
            self.update_weights()
    
    def update_weights(self):
        for key in self.W:
            self.W[key] -= self.learning_rate * self.dW[key]
            self.b[key] -= self.learning_rate * self.db[key]
    
    def predict(self, inputs):
        return self.forward(inputs)[0]
