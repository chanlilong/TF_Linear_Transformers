import torchvision
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf


class MNIST(Dataset):

    def __init__(self,):
        self.dataset = torchvision.datasets.MNIST(root="~/Handwritten_Deep_L/",train=True,download=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        x,y = self.dataset[idx]
        x=np.asarray(x,dtype=np.float32)/255.0
        y=np.asarray(y).reshape(-1,1)

        return x[...,None],y

def collate_fn_mnist(batch):
    image,label = zip(*batch)
    b = len(image)
    images=[]
    labels=[]
    for idx,(im,l) in enumerate(zip(image,label)):
        images.append(im.reshape(-1,1))
        labels.append(l)

    return tf.stack(images),tf.stack(labels)

def collate_fn_mnist_generative(batch):
    image,label = zip(*batch)

    images=[]
    labels=[]
    for (im,l) in (zip(image,label)):
        images.append(im.reshape(-1,1))
        labels.append(l.reshape(1))

    x= tf.stack(images)
    y = tf.stack(labels)
    return x[:,0:784//2,:],x[:,784//2:,:],y