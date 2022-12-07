# Linear Attention Transformers

- Linear Attention in this repo is through kernel approximation of the softmax function to avoid O(n^2) cost

$SM(X,Y) = \phi(X)\phi(Y)$
$\phi$ in this case is defined by two papers:
1. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
2. [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)

- Causal attention is also implemented through tf.cumsum and tf.einsum
- Training script for generating MNIST through transformers is added

```python
    import tensorflow as tf
    from Linear_Transformer import Linear_Transformer
    
    Q = tf.random.normal((1,512,256))
    K = tf.random.normal((1,784,256))
    model = Linear_Transformer(n_enc_layers=6,n_dec_layers=6,dim=256,n_heads=8,ffn_dim=1024,dropout=0.2,return_inter = False,causal=False)
    model(Q,K)
```