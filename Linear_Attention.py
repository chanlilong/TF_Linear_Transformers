import tensorflow as tf
from tensorflow.keras.layers import Dense
import math

@tf.function(jit_compile=True)
def project_feat_FAVOR(X,is_query=False):
    """
    Projects Tensor according to approximate softmax kernel.
    Args:
        X: query_prime tensor of the shape [B,L,H,M].
    """
    data_normalizer = 1.0 / (math.sqrt(math.sqrt(X.shape[-1])))
    X = data_normalizer*X
    W = tf.random.normal((1,X.shape[3],X.shape[3]))
    W_,_= tf.linalg.qr(W)
    W_ /= tf.linalg.norm(W_,axis=2)
    W_ *= math.sqrt(X.shape[3])#
    W_ = tf.repeat(W_,X.shape[0],axis=0)
    
    ratio = 1.0 / math.sqrt(W_.shape[1])
    
    diag_data = tf.square(X)
    diag_data = tf.reduce_sum(diag_data, axis=-1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = tf.expand_dims(diag_data, axis=-1)

    dot_prod = tf.einsum("bdm,blhm -> blhd",W_,X)
    if is_query:
        X = dot_prod - diag_data - tf.reduce_max(X, axis=-1, keepdims=True)#- tf.linalg.norm(X,axis=3)[...,None]**2/2
    else:
        X = dot_prod - diag_data - tf.reduce_max(X, axis=(-1,-3), keepdims=True)
    return ratio*tf.exp(X + 1e-08)
    
@tf.function(jit_compile=True)
def causal_attn(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.
  Args:
    qs: query_prime tensor of the shape [B,L,H,M].
    ks: key_prime tensor of the shape [B,L,H,M].
    vs: value tensor of the shape [B,L,H,D].
  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.

    WARNING:Initial Compilation will take 1min+
            after that its blazing
  """

  result = []
  result_norm = []
  B,I,J,K = ks.shape
  _,_,_,L = vs.shape
  sums = tf.zeros((B,J,K,L))
  sums_norm = tf.zeros((B,J,K))

  for index in range(qs.shape[1]):
    sums_norm = sums_norm  + ks[:,index]
    sums = sums + tf.einsum("ijk,ijl->ijkl", ks[:,index], vs[:,index])
    result.append(tf.einsum("ijkl,ijk->ijl", sums, qs[:,index])[None,...])
    result_norm.append(tf.reduce_sum(qs[:,index] * sums_norm, axis=2,keepdims=True)[None,...])

  result = tf.concat(result, axis=1)
  result_norm = tf.concat(result_norm, axis=1)

  return result/result_norm

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
                 d_values=None, d_model_keys=None, causal=False):
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
        # self.feat_proj = lambda x,is_query=False:tf.nn.elu(x)+1 #From "Transformers are RNNs"
        self.feat_proj = project_feat_FAVOR
        self.causal = causal

    def call(self, queries, keys, values):

        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = tf.reshape(self.query_projection(queries),(N, L, H, -1))
        keys = tf.reshape(self.key_projection(keys),(N, S, H, -1))
        values = tf.reshape(self.value_projection(values),(N, S, H, -1))

        Q_lin = self.feat_proj(queries,is_query=True)
        K_lin = self.feat_proj(keys)

        if self.causal:
            V_lin = causal_attn(Q_lin, K_lin, values)
        else:
            Z = 1/(tf.einsum("nlhd,nhd->nlh", Q_lin, tf.reduce_sum(K_lin,axis=1))+1e-10)

            KV = tf.einsum("nshd,nshm->nhmd", K_lin, values)

            V_lin = tf.einsum("nlhd,nhmd,nlh->nlhm", Q_lin, KV, Z)


        new_values = tf.reshape(V_lin,(N, L, -1))

        # Project the output and return
        return self.out_projection(new_values)

class FullAttentionLayer(tf.keras.Model):
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
        # self.feat_proj = lambda x:tf.nn.elu(x)+1 #From "Transformers are RNNs"
        # self.feat_proj = project_feat_FAVOR

    def call(self, queries, keys, values):

        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = tf.reshape(self.query_projection(queries),(N, L, H, -1))
        keys = tf.reshape(self.key_projection(keys),(N, S, H, -1))
        values = tf.reshape(self.value_projection(values),(N, S, H, -1))
        D = keys.shape[-1]
        sm_map = tf.einsum("nlhd,nshd->nlhs",queries,keys)
        sm_map = sm_map/math.sqrt(D)
        sm_map = tf.nn.softmax(sm_map,axis=1)
        attn = tf.einsum("nlhs,nshd->nlhd",sm_map,values)

        new_values = tf.reshape(attn,(N, L, -1))

        # Project the output and return
        return self.out_projection(new_values)