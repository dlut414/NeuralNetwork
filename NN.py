import numpy as np;
#import scipy as sp;

def sigmoid(z):
    return 1. / (1. + np.exp(-z));

def dSigmoid(z):
    sig = sigmoid(z);
    return sig* (1 - sig);

def ReLU(z):
    return np.maximum(np.zeros(z.shape), z);

def dReLU(z):
    return 1.0* (z >= 0);

def LReLU(z):
    return np.maximum(0.01*z, z);

def dLReLU(z):
    return 1.0* (z >= 0) + 0.01* (z < 0);

class Neuron:
    def __init__(self, nCur, nPre, g, dg, winit=0.1):
        self.nCur = nCur;
        self.nPre = nPre;
        self.w = winit* np.random.randn( nCur, nPre );
        self.b = np.zeros( (nCur,1) );
        self.g = g;
        self.dg = dg;
    def forwardProp(self, x):
        self.z = np.matmul(self.w, x) + self.b;
        self.a = self.g(self.z);
    def backwardProp(self, Da, a_left):
        self.Dz = Da* self.dg(self.z);
        self.Dw = np.matmul(self.Dz, a_left.transpose());
        self.Db = np.sum(self.Dz, axis=1, keepdims=True);
        self.Da_left = np.matmul(self.w.transpose(), self.Dz);

class NN:
    def __init__(self, Layers, alpha=0.03, reg=0):
        self.sigma = 1e-15;
        self.Layers = Layers;
        self.alpha = alpha;
        self.reg = reg;
        self.Layers = Layers;
        self.nLayer = len(Layers);
        self.Neurons = [];
        self.Neurons.append( Neuron(Layers[0], 1, ReLU, dReLU) );
        for i in range(1, self.nLayer-1):
            self.Neurons.append( Neuron(Layers[i], Layers[i-1], ReLU, dReLU) );
        self.Neurons.append( Neuron(Layers[-1], Layers[-2], sigmoid, dSigmoid) );

    def J(self, a, y):
        m = y.shape[1];
        a = np.maximum(a, self.sigma);
        a = np.minimum(a, 1-self.sigma);
        return ( np.inner(y, np.log(a)) + np.inner(1-y, np.log(1-a)) )/(-m);

    def dJ(self, a, y):
        m = y.shape[1];
        a = np.maximum(a, self.sigma);
        a = np.minimum(a, 1-self.sigma);
        return (a - y) / (a*(1 - a)) / m;

    def predict(self, x):
        self.Neurons[1].forwardProp( x );
        for i in range(2, self.nLayer):
            self.Neurons[i].forwardProp( self.Neurons[i-1].a );
        return self.Neurons[-1].a;

    def predict01(self, x):
        return self.predict(x) > 0.5;

    def dPredict(self, Da, x):
        if self.nLayer == 2:
            self.Neurons[-1].backwardProp(Da, x);
        else:
            self.Neurons[-1].backwardProp(Da, self.Neurons[-2].a);
            for i in range(self.nLayer-2,1,-1):
                self.Neurons[i].backwardProp(self.Neurons[i+1].Da_left, self.Neurons[i-1].a);
            self.Neurons[1].backwardProp(self.Neurons[2].Da_left, x);
        return;

    def train_once(self, x, y):
        m = y.shape[1];
        a = self.predict(x);
        self.dPredict(self.dJ(a, y), x);
        for j in range(1, self.nLayer):
            self.Neurons[j].w -= self.alpha* self.Neurons[j].Dw + self.reg* self.Neurons[j].w / m;
            self.Neurons[j].b -= self.alpha* self.Neurons[j].Db;
        return;

    def test(self, x, y):
        m = y.shape[1];
        a = self.predict01(x);
        self.accuracy = np.sum(a == y) / float(m);
        nOne = np.sum(y, axis=1);
        nPos = np.sum(a, axis=1);
        self.precision = 0.0;
        if nOne > 0:
            self.precision = np.sum( np.logical_and(a == 1, y == 1) ) / float(nOne);
        self.recall = 0.0;
        if nPos > 0:
            self.recall = np.sum( np.logical_and(a == 1, y == 1) ) / float(nPos);
        self.fscore = 0.0;
        if self.precision + self.recall > 0:
            self.fscore = 2*self.precision*self.recall / (self.precision + self.recall);
        return;

    def save(self, filename):
        import json;
        data = {};
        data.update({'nLayer':self.nLayer, 'alpha':self.alpha, 'reg':self.reg});
        data.update({'Layers':self.Layers});
        for i in range(0, self.nLayer):
            data.update({'w'+str(i):self.Neurons[i].w.tolist()});
            data.update({'b'+str(i):self.Neurons[i].b.tolist()});
            data.update({'fun'+str(i):self.Neurons[i].g.__name__});
        with open(filename, 'w') as json_data:
            json.dump(data, json_data);
        return;

    def load(self, filename):
        import json;
        with open(filename, 'r') as json_data:
            data = json.load(json_data);
            self.__init__(data['Layers'], data['alpha'], data['reg']);
            for i in range(0, data['nLayer']):
                self.Neurons[i].w = np.array(data['w'+str(i)], ndmin=2);
                self.Neurons[i].b = np.array(data['b'+str(i)], ndmin=2);
        return;
