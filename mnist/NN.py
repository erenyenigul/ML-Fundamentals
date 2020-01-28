
import numpy as np


class NN :

    def __init__(self):
        W1 ,b1 ,W2 ,b2 = 0 ,0 ,0 ,0

    def loss(self, X, y ,reg):
        num_train = X.shape[0]
        X -= np.mean(X, axis=0, keepdims=True)

        ReLu = lambda x: np.maximum(x.copy(), 0, x.copy())

        X1 = X.dot(self.W1) + self.b1
        X2 = ReLu(X1)
        X3 = X2.dot(self.W2) + self.b2
        softmax = np.exp(X3) / np.sum(np.exp(X3), axis=1, keepdims=True)

        loss = -np.log(softmax[range(len(softmax)), y])
        loss = np.mean(loss)

        loss += np.sum(self.W1 ) *reg
        loss += np.sum(self.W2 ) *reg
        loss += np.sum(self.b1 ) *reg
        loss += np.sum(self.b2 ) *reg

        softmax[range(num_train) ,y] -=1

        dW2 = X2.T.dot(softmax) / num_train
        dW2 += 2* self.W2 * reg

        db2 = np.sum(softmax, axis=0,keepdims=True) / num_train
        db2 += 2 * self.b2 * reg

        dW1 = softmax.dot(self.W2.T)
        dX2 = dW1 * (X1 > 0)
        dW1 = X.T.dot(dX2) / num_train
        dW1 += 2 * self.W1 * reg

        db1 = np.sum(dX2, axis=0,keepdims=True) / num_train
        db1 += 2 * self.b1 * reg

        grads = {'dW2': dW2, 'dW1': dW1, 'db2': db2, 'db1': db1}

        return loss, grads

    def train(self, X_train, y_train, iterations=1000, reg=0.00001, learning_rate=0.001, batch_size=800):
        num_train, dim = X_train.shape
        num_classes = np.max(y_train) + 1

        self.W1 = 0.001 * np.random.randn(dim, 100)
        self.b1 = 0.001 * np.random.randn(1, 100)
        self.W2 = 0.001 * np.random.randn(100, num_classes)
        self.b2 = 0.001 * np.random.randn(1, num_classes)

        for i in range(iterations):
            batch_index = np.random.choice(num_train, batch_size)
            X_batch = X_train[batch_index]
            y_batch = y_train[batch_index]

            loss, grads = self.loss(X_batch, y_batch, reg)

            self.W1 -= grads['dW1'] * learning_rate
            self.b1 -= grads['db1'] * learning_rate
            self.W2 -= grads['dW2'] * learning_rate
            self.b2 -= grads['db2'] * learning_rate
            print(loss)

    def predict(self, X):
        ReLu = lambda x: np.maximum(x.copy(), 0, x.copy())

        X1 = X.dot(self.W1) + self.b1
        X2 = ReLu(X1)
        X3 = X2.dot(self.W2) + self.b2

        return np.argmax(X3, axis=1)

