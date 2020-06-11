from SIT import *
import os
from load_data import load_data_celeba

data_train = load_data_celeba(flag='training', side_length=64)

path = 'AE_CelebA_latent64'
generator_path   = os.path.join(path,'decoder')
encoder_path     = os.path.join(path,'encoder')

latent_dim = 64

tf.reset_default_graph()

data          = tf.placeholder(shape=[16, 64, 64, 3],dtype=tf.float32)
latent_data   = tf.placeholder(shape=[None,latent_dim], dtype=tf.float32)
encoder       = hub.Module(encoder_path, trainable=False)
decoder       = hub.Module(generator_path, trainable=False)

encoded, _    = tf.split(encoder({'x':data},as_dict=True)['z'], 2, axis=-1)
reconstruct   = decoder({'z':encoded},as_dict=True)['x']
decoded       = decoder({'z':latent_data},as_dict=True)['x']

sess = tf.Session()
sess.run(tf.global_variables_initializer())

chunk_size = 16
start = 0
end = chunk_size
data0 = torch.zeros(data_train.shape[0], latent_dim)
while(True):
    temp = sess.run(encoded, feed_dict={data:data_train[start:end].reshape(end-start,64,64,3).astype(float)/255.0-0.5})
    data0[start:end] = torch.tensor(temp)

    start = end
    end = end+chunk_size
    if start >= data_train.shape[0]:
        break
    if end > data_train.shape[0]:
        end = data_train.shape[0]
        start = end - chunk_size

data_train = data0


KDE = False
nsample_wT = 80000 
nsample_spline = 80000
nsample = nsample_wT + nsample_spline 

batchsize = 20000 #to avoid running out of memory
ndim_latent = 64 

t_total = time.time()

#define the model
model = SIT(ndim=ndim_latent).requires_grad_(False).cuda()

sample = torch.randn(nsample, ndim_latent)

#Add 200 regular layers
for i in range(500):
    if len(model.layer) < 100:
        n_component = 6 
    elif len(model.layer) < 200:
        n_component = 4
    else:
        n_component = 2 

    model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', KDE=KDE, batchsize=batchsize)

print (time.time()-t_total) 
torch.save(model, 'SIG_CelebA_AE64')
