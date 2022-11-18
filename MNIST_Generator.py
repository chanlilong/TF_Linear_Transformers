from Linear_Transformer import Linear_Transformer
import tensorflow as tf
from tensorflow.keras.layers import Dense


class MNIST_Generative_Model(tf.keras.Model):

    def __init__(self,return_inter=False):
        super().__init__()
        self.transformer = Linear_Transformer(return_inter=return_inter)
        self.emb = tf.Variable(initial_value=tf.zeros((1,784//2,256)),trainable=True,name="query_emb")
        self.linear_in = Dense(256)
        self.linear_out = Dense(1,activation = tf.nn.sigmoid)
        self.return_inter=return_inter
    def call(self,x):
        query = tf.repeat(self.emb,x.shape[0],axis=0)
        x = self.linear_in(x)
        x = self.transformer(query,x)
        if self.return_inter:
            x = [self.linear_out(i) for i in x]
        else:
            x = self.linear_out(x)
        return x