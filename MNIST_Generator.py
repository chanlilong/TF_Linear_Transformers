from Linear_Transformer import Linear_Transformer
import tensorflow as tf
from tensorflow.keras.layers import Dense


class MNIST_Generative_Model(tf.keras.Model):

    def __init__(self,dim=256,return_inter=False):
        super().__init__()
        self.transformer = Linear_Transformer(dim=dim,return_inter=return_inter)
        self.emb = tf.Variable(initial_value=tf.zeros((1,784//2,dim//2)),trainable=True,name="query_emb")
        self.linear_in = Dense(dim)
        self.linear_out = Dense(1,activation = tf.nn.sigmoid)
        self.return_inter=return_inter
        self.int_emb = tf.keras.layers.Embedding(10,dim//2)
    def call(self,x,y):

        int_emb = self.int_emb(y)[:,None,:]
        int_emb = tf.repeat(int_emb,784//2,axis=1)
        query = tf.repeat(self.emb,x.shape[0],axis=0)
        query = tf.concat([query,tf.squeeze(int_emb,axis=2)],axis=2)#[16,392,128], [16,392,1,128]

        x = self.linear_in(x)
        x = self.transformer(query,x)
        if self.return_inter:
            x = [self.linear_out(i) for i in x]
        else:
            x = self.linear_out(x)
        return x