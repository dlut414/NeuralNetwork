import numpy as np;
from NN import NN as NN;
#import gradientCheck as gc;
# hyperparameters, functions

Layers = (2, 16, 16, 1);
maxIteration = 2000;
alpha = 0.03;
reg = 0;
batch_size = 100;

nn = NN(Layers, alpha, reg);

m = 10000;
xt = 2.0*np.random.random((m, 2)) - 1.0;
np.random.shuffle(xt);
x = xt.transpose();
q = np.logical_xor(x[0,:] > 0, x[1,:] > 0);
y = q.reshape((1,m));

m_train = int(0.7*m);
x_train = x[:,0:m_train];
y_train = y[:,0:m_train];
x_val = x[:,m_train:];
y_val = y[:,m_train:];


n_batch = m_train / batch_size;
for epoch in range(0, maxIteration):
    for i in range(0, n_batch):
        start = i* batch_size;
        end = np.minimum(start+batch_size, m_train);
        nn.train_once(x_train[:,start:end], y_train[:,start:end]);
    print epoch, " train cost J: ", np.asscalar(nn.J(nn.predict(x), y));
    print epoch, " test cost J: ", np.asscalar(nn.J(nn.predict(x_val), y_val));
nn.test(x_val, y_val);

print " accuracy: ", nn.accuracy;
print " precision: ", nn.precision;
print " recall: ", nn.recall;
print " fScore: ", nn.fscore;
        
'''
a = predict(x);
dPredict(dJ(a,y), x);
gc.gradientCheck(Neurons, predict, J, x, y);
'''
