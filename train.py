from MNIST_Generator import MNIST_Generative_Model
from mnist_dataset import collate_fn_mnist_generative,MNIST
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
EPOCHS =5
N_BATCH = 16

if __name__=="__main__":
    writer = SummaryWriter(logdir = "./tensorboard_logs/mnist_generator2")

    model = MNIST_Generative_Model(return_inter=True)
    dataset = MNIST()
    dataloader = DataLoader(dataset,batch_size=N_BATCH,shuffle=True,num_workers=4,collate_fn=collate_fn_mnist_generative)
    opt = tf.keras.optimizers.Adam(lr=1e-04)
    # t_vars = model.trainable_variables
    itr=0
    for e in (range(EPOCHS)):
        for x,y in tqdm(dataloader):
            
            with tf.GradientTape() as tape:

                pred = model(x)
                loss = sum([tf.reduce_mean(tf.square(p-y)) for p in pred])
            
            dl_dw = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(dl_dw, model.trainable_variables))
            
            if itr%2==0:
                writer.add_scalar("train_losses/loss",loss.numpy().item()/6,itr)
            if itr%10==0:
                pred = model(x[0:1],training=False)[-1]
                pred_img = tf.stack([x[0:1],pred],axis=1).numpy()
                pred_img = pred_img.reshape(28,28,1)

                fig,ax = plt.subplots(1,1,figsize=(8,8))
                ax.imshow(pred_img)
                writer.add_figure("images/pred",fig,itr)

            if itr%100==0 and itr!=0:
                model.save_weights("./model.h5")

            itr+=1
    