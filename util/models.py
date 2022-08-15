from util import *
from util.plotting import plot_points_and_vocab
from tqdm import tqdm
import numpy as np
import string
import sys


class skipgram():
    """
    Vanilla Skip-gram model. 
    Loss:
        Cross-entropy loss H(P, Q) with P as target one-hot vector.
    Optimization:
        Stochastic Gradient Descent.
    """
    def __init__(self, 
                 dataset):
        """
        """
        self.X_train = []
        self.y_train = []
        self.alpha = learning_rate
        self.vocabulary = set()
        
        # Construct vocabulary
        for doc in dataset:
            for word in doc:
                self.vocabulary.add(word)
        self.vocabulary = list(self.vocabulary)
        
        # Construct training dataset
        for doc in dataset:
            for i in range(len(doc)):
                context_word = [0] * len(self.vocabulary)
                target_word = [0] * len(self.vocabulary)
                
                context_word[self.vocabulary.index(doc[i])] = 1
                
                for j in range(i - window_size, i + window_size):
                    if i != j and j >= 0 and j < len(doc):
                        target_word[self.vocabulary.index(doc[j])] = 1
                        self.X_train.append(context_word)
                        self.y_train.append(target_word)
        
        # Construct weights
        self.U = np.random.uniform(low=initialization_low, 
                                   high=initialization_high, 
                                   size=(len(self.vocabulary), embedding_dim))
        self.V = np.random.uniform(low=initialization_low, 
                                   high=initialization_high, 
                                   size=(embedding_dim, len(self.vocabulary)))
        
    @staticmethod
    def softmax(x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x/e_x.sum(), 1/e_x.sum()
        
    def feed_forward(self,
                     X):
        """
        """
        self.y, self.A = self.softmax(np.dot(self.V.T, self.U[X.index(1)]))
        
    def train(self):
        """
        """
        for epoch in tqdm(range(1, train_epochs), position=0):
            loss = 0
            for j in range(len(self.X_train)):
                self.feed_forward(self.X_train[j])
                old_u_c = self.U[self.X_train[j].index(1)]
                old_v_w = self.V.T[self.y_train[j].index(1)]
                
                # Update weights U, V
                self.U[self.X_train[j].index(1)] -= self.alpha * (self.A * sum([v * np.exp(np.dot(old_u_c, v)) for v in self.V.T]) - old_v_w)
                
                for k in range(len(self.vocabulary)):
                    if self.y_train[j].index(1) != k:
                        self.V.T[k] -= self.alpha * (self.A * np.exp(np.dot(old_u_c, self.V.T[k])) * old_u_c)
                    else:
                        self.V.T[k] = old_v_w - self.alpha * (self.A * np.exp(np.dot(old_u_c, old_v_w)) - 1) * old_u_c

                loss -= np.log(np.exp(np.dot(self.U[self.X_train[j].index(1)], 
                                             self.V.T[self.y_train[j].index(1)])) * self.softmax(np.dot(self.V.T, self.U[self.X_train[j].index(1)]))[1])
                

            tqdm.write("Epoch: " + str(epoch) + ", Loss: " + str(loss/len(self.X_train)))
            if epoch % save_iteration == 0:
                self.alpha *= 1/( (1 + self.alpha * epoch) )
                self.save_checkpoint(epoch)
            
            # Plot
            plot_points_and_vocab(self.U, 
                                  self.vocabulary, 
                                  embedding_dim,
                                  save_plots=True,
                                  fig_show=False,
                                  name=str(epoch))
            
    def save_checkpoint(self, 
                        epoch):
        """
        """
        with open(checkpoint_dir + "checkpoint-" + str(epoch) + "-U.npy", 'wb') as f:
            np.save(f, self.U)
        with open(checkpoint_dir + "checkpoint-" + str(epoch) + "-V.npy", 'wb') as f:
            np.save(f, self.V)
            
    def predict(self,
                word,
                k):
        """
        """
        if word in self.vocabulary:
            onehot = [0] * len(self.vocabulary)
            onehot[self.vocabulary.index(word)] = 1

            self.feed_forward(onehot)
            
            k_closest_words = sorted(range(len(self.y)), key=lambda i: self.y[i])[-k-1:]
            
            words = [self.vocabulary[i] for i in k_closest_words]
            try:
                words.remove(word)
            except ValueError:
                words = words[:-1]
            return words
            
        else:
            print("Word not in dictionary")