from MNIST_Generator import MNIST_Generative_Model
from mnist_dataset import collate_fn_mnist_generative,MNIST
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
import seaborn as sns


EPOCHS =5
N_BATCH = 16

if __name__=="__main__":
    # LOSS = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.NONE)#reduction=tf.keras.losses.Reduction.NONE
    
    @tf.function
    def train_mnist(x1,x2,y):

        with tf.GradientTape() as tape:
            pred,_ = model(x1,y)
            # loss = sum([tf.reduce_mean(LOSS(x2,p)) for p in pred])
            loss = sum([tf.reduce_mean(0.75*tf.square(x2-p)+0.25*tf.abs(x2-p)) for p in pred])

        dl_dw = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(dl_dw, model.trainable_variables))

        return loss

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    writer = SummaryWriter(logdir = "./tensorboard_logs/mnist_generator_linearattnL1")

    model = MNIST_Generative_Model(dim=64,return_inter=True)
    dataset = MNIST()
    dataloader = DataLoader(dataset,batch_size=N_BATCH,shuffle=True,collate_fn=collate_fn_mnist_generative)
    opt = tf.keras.optimizers.Adam(lr=1e-04)
    # t_vars = model.trainable_variables
    itr=0
    for e in (range(EPOCHS)):
        for x1,x2,y in tqdm(dataloader):
            
            loss = train_mnist(x1,x2,y)

            itr+=1
            
            if itr%2==0:
                writer.add_scalar("train_losses/loss",loss.numpy().item()/6,itr)
            if itr%100==0:
                pred,qk = model(x1[0:1],y[0:1],True,training=False)
                pred = pred[-1]
                # print(qk[0].shape,qk[1].shape)
                # qk = tf.reduce_mean(tf.einsum("nlhd,nthd->nlht",qk[0],qk[1]),axis=2)
                pred_img = tf.stack([x1[0:1],pred],axis=1).numpy()
                pred_img = pred_img.reshape(28,28,1)

                fig,ax = plt.subplots(1,1,figsize=(8,8))
                ax.imshow(pred_img)
                
                writer.add_figure("images/pred",fig,itr)
                plt.close()

                fig,ax = plt.subplots(1,1,figsize=(8,8))
                sns.heatmap(qk[0].numpy(),ax=ax)
                writer.add_figure("images/attention",fig,itr)
                plt.close()
        model.save_weights("./model.h5")
model.save_weights("./model.h5")