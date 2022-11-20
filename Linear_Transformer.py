import tensorflow as tf
import math
from tensorflow.keras.layers import Dense,LayerNormalization,Dropout
from tensorflow.keras import Sequential
from Linear_Attention import LinearAttentionLayer,FullAttentionLayer
import keras_nlp

class Linear_Transformer_Encoder(tf.keras.Model):
    
    def __init__(self,n_enc_layers=6,dim=256,n_heads=8,ffn_dim=1024,dropout=0.2):
        super().__init__()

        # self.MHA = [FullAttentionLayer(d_model=dim, n_heads=n_heads , d_keys=dim,d_values=dim,d_model_keys=dim) for _ in range(n_enc_layers)] #18GB
        self.MHA = [LinearAttentionLayer(d_model=dim, n_heads=n_heads , d_keys=dim,d_values=dim,d_model_keys=dim) for _ in range(n_enc_layers)]
        self.norm1 = [LayerNormalization(axis=-1) for _ in range(n_enc_layers)]
        self.norm2 = [LayerNormalization(axis=-1) for _ in range(n_enc_layers)]
        self.do1 = [Dropout(dropout) for _ in range(n_enc_layers)]
        self.FFN = [Sequential(layers=[Dense(ffn_dim,activation=tf.nn.relu),\
                               Dropout(dropout),\
                               Dense(dim,activation=tf.nn.relu)]) for _ in range(n_enc_layers)]
        self.pos_enc = keras_nlp.layers.SinePositionEncoding()
    def call(self,x):
        positional_encoding = self.pos_enc(x)
        x = x+positional_encoding

        for mha,do,norm,ffn,norm2 in zip(self.MHA,self.do1,self.norm1,self.FFN,self.norm2):
            
            attn1 = mha(x,x,x)
            attn1 = do(attn1)
            x = norm(attn1+x)
            f = ffn(x)
            x = norm2(f+x)
            
        return x
            
class Linear_Transformer_Decoder(tf.keras.Model):
    
    def __init__(self,n_dec_layers=6,dim=256,n_heads=8,ffn_dim=1024,dropout=0.2,return_inter = False):
        super().__init__()
        # self.MHA1 = [FullAttentionLayer(d_model=dim, n_heads=n_heads , d_keys=dim,d_values=dim,d_model_keys=dim) for _ in range(n_dec_layers)]
        # self.MHA2 = [FullAttentionLayer(d_model=dim, n_heads=n_heads , d_keys=dim,d_values=dim,d_model_keys=dim) for _ in range(n_dec_layers)]
        self.MHA1 = [LinearAttentionLayer(d_model=dim, n_heads=n_heads ,d_keys=dim,d_values=dim,d_model_keys=dim) for _ in range(n_dec_layers)]
        self.MHA2 = [LinearAttentionLayer(d_model=dim, n_heads=n_heads ,d_keys=dim,d_values=dim,d_model_keys=dim) for _ in range(n_dec_layers)]
        self.norm1 = [LayerNormalization(axis=-1) for _ in range(n_dec_layers)]
        self.norm2 = [LayerNormalization(axis=-1) for _ in range(n_dec_layers)]
        self.norm3 = [LayerNormalization(axis=-1) for _ in range(n_dec_layers)]
        self.do1 = [Dropout(dropout) for _ in range(n_dec_layers)]
        self.do2 = [Dropout(dropout) for _ in range(n_dec_layers)]
        self.FFN = [Sequential(layers=[ Dense(ffn_dim,activation=tf.nn.relu),\
                                        Dropout(dropout),\
                                        Dense(dim,activation=tf.nn.relu)]) for _ in range(n_dec_layers)]
        self.pos_enc = keras_nlp.layers.SinePositionEncoding()
        self.return_intermediate = return_inter
    def call(self,Q,K):
        positional_encoding = self.pos_enc(Q)
        x1 = Q+positional_encoding

        if self.return_intermediate:
            feat_list = []
        for mha1,do1,norm1,mha2,do2,norm2,ffn,norm3 \
            in zip(self.MHA1,self.do1,self.norm1,self.MHA2,self.do2,self.norm2,self.FFN,self.norm3):
            
            x = mha1(x1,x1,x1)
            x = do1(x)
            x2 = norm1(x+x1)
            
            x = mha2(x2,K,K)
            x = do2(x)
            x3 = norm2(x+x2)
            
            x = ffn(x3)
            x1 = norm3(x+x3)

            if self.return_intermediate:
                feat_list.append(x1)

        if self.return_intermediate:
            return feat_list
        else:
            return x1        
    
class Linear_Transformer(tf.keras.Model):
    def __init__(self,n_enc_layers=6,n_dec_layers=6,dim=256,n_heads=8,ffn_dim=1024,dropout=0.2,return_inter = False):
        super().__init__()
        self.enc = Linear_Transformer_Encoder(n_enc_layers=n_enc_layers,dim=dim,n_heads=n_heads,ffn_dim=ffn_dim,dropout=dropout)
        self.dec = Linear_Transformer_Decoder(n_dec_layers=n_dec_layers,dim=dim,n_heads=n_heads,ffn_dim=ffn_dim,dropout=dropout,return_inter=return_inter)
        self.return_inter = return_inter
        
    def call(self,Q,K):
        
        Q_ = self.enc(Q)
        x  = self.dec(Q_,K)
        
        return x