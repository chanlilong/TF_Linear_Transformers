import tensorflow as tf
from tensorflow.keras.layers import Dense

class LinearAttentionLayer(tf.keras.Model):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.
    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.
    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super().__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.query_projection = Dense(d_keys * n_heads)
        self.key_projection = Dense(d_keys * n_heads)
        self.value_projection = Dense(d_values * n_heads)
        self.out_projection = Dense(d_model)
        self.n_heads = n_heads
        self.feat_proj = lambda x:tf.nn.elu(x)+1

    def call(self, queries, keys, values):

        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = tf.reshape(self.query_projection(queries),(N, L, H, -1))
        keys = tf.reshape(self.key_projection(keys),(N, S, H, -1))
        values = tf.reshape(self.value_projection(values),(N, S, H, -1))

        Q_lin = self.feat_proj(queries)
        K_lin = self.feat_proj(keys)

        Z = 1/(tf.einsum("nlhd,nhd->nlh", Q_lin, tf.reduce_sum(K_lin,axis=1))+1e-10)

        KV = tf.einsum("nshd,nshm->nhmd", K_lin, values)

        V_lin = tf.einsum("nlhd,nhmd,nlh->nlhm", Q_lin, KV, Z)

        new_values = tf.reshape(V_lin,(N, L, -1))

        # Project the output and return
        return self.out_projection(new_values)