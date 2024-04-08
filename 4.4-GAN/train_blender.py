import os
import time 
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import imageio 
import vgg_loss
import discriminators_pix2pix
import res_unet
import gan_loss
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils
from discriminators_pix2pix import MultiscaleDiscriminator
from res_unet import MultiScaleResUNet
from vgg_loss import VGGLoss
from gan_loss import GANLoss
from SwappedDataset import SwappedDatasetLoader   
from img_utils import tensor2rgb
from img_utils import tensor2bgr
from img_utils import rgb2tensor
from google.colab.patches import cv2_imshow

# Configurations ----> careful with data_root, root and small_root
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'Blender'
data_root = '/content/drive/MyDrive/Colab Notebooks/cv2_2022_assignment3/gan_blender_release/data_set/data/'
root = '/content/drive/MyDrive/Colab Notebooks/cv2_2022_assignment3/gan_blender_release/'

# for some small testing
small_root = '/content/drive/MyDrive/Colab Notebooks/cv2_2022_assignment3/gan_blender_release/dataset2/small_dataset/'
train_list = ''
test_list = ''
batch_size = 8
nthreads = 4
max_epochs = 1
displayIter = 1000
saveIter = 1
img_resolution = 256
lr_gen = 1e-4
lr_dis = 1e-4
momentum = 0.9
weightDecay = 1e-4

# they say in assignment step size 30 but then every 30 iterations the learning rate goes down
# meaning that in the first epoch the learning goes down to 1e-20, not desirable
# one epoch for batch size 8 has ca. 4500 iterations
step_size = 300000
gamma = 0.1
pix_weight = 0.1
rec_weight = 1.0
gan_weight = 0.001

######################################################################
# Independent code. Don't change after this line. All values are automatically
# handled based on the configuration part.

if batch_size < nthreads:
    nthreads = batch_size
    
check_point_loc = 'Exp_%s/checkpoints/' % experiment_name.replace(' ', '_')
visuals_loc = 'Exp_%s/visuals/' % experiment_name.replace(' ', '_')
os.makedirs(check_point_loc, exist_ok=True)
os.makedirs(visuals_loc, exist_ok=True)
checkpoint_pattern = root+'Blender/checkpoints/checkpoint_%s_%d.pth'
logTrain = root+'Blender/LogTrain.txt'

torch.backends.cudnn.benchmark = True
cudaDevice = ''

flag = 0
if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else: 
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

print(torch.cuda.get_device_name(torch.cuda.current_device()))

done = u'\u2713'
print('[I] STATUS: Initiate Network and transfer to device...', end='')
dis = MultiscaleDiscriminator().to(device)
gen = MultiScaleResUNet(in_nc=7,out_nc=3).to(device) 
print(done)

print('[I] STATUS: Initiate optimizer...', end='')
# Define your optimizers and the schedulers and connect the networks from
gen_optimizer = torch.optim.SGD(gen.parameters(), lr=1e-4, momentum=momentum, weight_decay=weightDecay)
gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=step_size, gamma=gamma)

dis_optimizer = torch.optim.SGD(dis.parameters(), lr=1e-4, momentum=momentum, weight_decay=weightDecay)
dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer, step_size=step_size, gamma=gamma)

print('[I] STATUS: Load Networks...', end='')
# Load your pretrained models here. Pytorch requires you to define the model
# before loading the weights, since the weight files does not contain the model
# definition. Make sure you transfer them to the proper training device. Hint:
# use the .to(device) function, where device is automatically detected
# above.

# when using CPU somehow the thing doesnt work so change it to this otherise use the other provided one
def loadModels2(model, path, optims=None, Test=True, device=None):
    checkpoint = torch.load(path, map_location={'cuda:0':device})
    model.load_state_dict(checkpoint['model'])
    if not Test:
        optims.load_state_dict(checkpoint['optimizer'])
    return model, optims, checkpoint['iters']

# load on gpu. can not be loaded on TPU
if flag != 1:
    gen, gen_optims, gen_iter_checkpoint = utils.loadModels(gen, root+'Blender/checkpoints/checkpoint_G.pth', 
                                                            optims=None, Test=True)
    dis, dis_optims, dis_iter_checkpoint = utils.loadModels(dis, root+'Blender/checkpoints/checkpoint_D.pth', 
                                                            optims=None, Test=True) 
# load on cpu  
else:
    gen, gen_optims, gen_iter_checkpoint = loadModels2(gen, root+'Blender/checkpoints/checkpoint_G.pth', 
                                                optims=gen_optimizer, Test=False, device=device)
    dis, dis_optims, dis_iter_checkpoint = loadModels2(dis, root+'Blender/checkpoints/checkpoint_D.pth', 
                                                    optims=dis_optmizer, Test=False, device= device)     
print(done)

print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
# Define your criterions here and transfer to the training device. They need to
# be on the same device type.
criterion_pixelwise = torch.nn.L1Loss().to(device)
criterion_id = VGGLoss().to(device)
criterion_gan = GANLoss(use_lsgan=True).to(device)
print(done) 

print('[I] STATUS: Initiate Dataloaders...')
dataset_train =  SwappedDatasetLoader(data_root, small_root, train=True, transform=None) 
train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True,num_workers=4, 
                            pin_memory=True, drop_last=True)
print(done)

print('[I] STATUS: Initiate Logs...', end='')
trainLogger = open(logTrain, 'w')
print(done)

# train FSGAN. Reference: github source code FSGAN
def Train(G, D, epoch_count, iter_count, trainLogger):
    G.train(True)
    D.train(True)
    epoch_count += 1
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    Epoch_time = time.time()
    total_loss_D = []
    total_loss_D_fake = []
    total_loss_D_true = []
    total_loss_G = []
    total_loss_G_only = []
    total_loss_pix = []
    total_loss_id = []

    for i, data in pbar:

        iter_count += 1
        images = data

        #### all the blending is placed in the dataloader for convenience
        try:
            source = images['source'].to(device).squeeze()  
            target = images['target'].to(device).squeeze()
            swap = images['swap'].to(device).squeeze()
            mask = images['mask'].to(device).squeeze()
            groundtruth = images['gr'].to(device).squeeze()
            img_transfer = images['transfer'].to(device).squeeze()
        except:
            print("correpted input tuple")
            continue

        ##### GENERATOR #######
        gen_optimizer.zero_grad()
        
        # create the concateated input (2 color, 1 grey image)
        m = mask[:,:1,:,:].to(device)
        gen_input = torch.cat((img_transfer, target, m), dim=1)
        generated_sample = G([gen_input]) 
        
        pred_fake = D([generated_sample]) 
        loss_G_GAN = criterion_gan(pred_fake, True)  

        loss_pixelwise = criterion_pixelwise(generated_sample, groundtruth)
        loss_id = criterion_id(generated_sample, groundtruth)
        loss_rec = pix_weight * loss_pixelwise + loss_id 
        loss_G_total = rec_weight * loss_rec + 0.001 * loss_G_GAN
        
        loss_G_total.backward()
        gen_optimizer.step()
        gen_scheduler.step()
        
        # save for mean over epoch G and rec losses
        total_loss_G_only.append(loss_G_GAN.detach().cpu().numpy())
        total_loss_G.append(loss_G_total.detach().cpu().numpy())
        total_loss_pix.append(loss_pixelwise.detach().cpu().numpy())
        total_loss_id.append(loss_id.detach().cpu().numpy()) 

        ##### DISCRIMINATOR #######
        dis_optimizer.zero_grad()

        # detach generated sample otherwise wrong gradient
        pred_fake = D([generated_sample.detach()]) 
        loss_D_fake = criterion_gan(pred_fake, False)
        
        pred_real = D([target])
        loss_D_real = criterion_gan(pred_real, True)
        loss_D_total = 0.5* loss_D_fake + 0.5* loss_D_real
        
        loss_D_total.backward()
        dis_optimizer.step()
        dis_scheduler.step()
        
        # save for mean over epoch D losses
        total_loss_D.append(loss_D_total.detach().cpu().numpy())
        total_loss_D_fake.append(loss_D_fake.detach().cpu().numpy())
        total_loss_D_true.append(loss_D_real.detach().cpu().numpy())

        # save images, there is a prettier way to do this using the provided grid function
        if i % displayIter == 0:
            for j in range(8):

                cv2.imwrite(root+'visuals/Epoch_'+str(j)+'_'+str(epoch_count)+'_iteration_'+str(i)+'_source.png', tensor2rgb(source[0].detach()))

                cv2.imwrite(root+'visuals/Epoch_'+str(j)+'_'+str(epoch_count)+'_iteration_'+str(i)+'_target.png', tensor2rgb(target[j].detach()) )

                cv2.imwrite(root+'visuals/Epoch_'+str(j)+'_'+str(epoch_count)+'_iteration_'+str(i)+'_swap.png', tensor2rgb(swap[0].detach()))
                
                cv2.imwrite(root+'visuals/Epoch_'+str(j)+'_'+str(epoch_count)+'_iteration_'+str(i)+'_img_blend.png', tensor2rgb(groundtruth[j].detach()))
                
                cv2.imwrite(root+'visuals/Epoch_'+str(j)+'_'+str(epoch_count)+'_iteration_'+str(i)+'_img_blend_pred.png',tensor2rgb(generated_sample[j].detach()))
                
                cv2.imwrite(root+ 'visuals/Epoch_'+str(j)+'_'+str(epoch_count)+'_iteration_'+str(i)+'_mask.png',tensor2rgb(mask[0].detach()))

        # log the losses
        trainLogger.write('EPOCH: '+str(epoch_count)+'/'+str(max_epochs)+'\n'
                        'iteration:'+str(i)+'\n'
                        'G_loss_total:'+str(loss_G_total)+'\n'
                        'G_loss:'+str(loss_G_GAN)+'\n'
                        'PIX_loss:'+str(loss_pixelwise)+'\n'                        
                        'ID_loss:'+str(loss_id)+'\n'
                        'loss_D_total:'+str(loss_D_total)+'\n'                        
                        'loss_D_fake:'+str(loss_D_fake)+'\n'  
                        'loss_D_real:'+str(loss_D_real)+'\n'                          
        ) 
        
        # print the losses in pbar
        pbar.set_postfix(lossG=loss_G_total.item(),lossD=loss_D_total.item(), lossDFake=loss_D_fake.item(), G=loss_G_GAN.item())
        pbar.set_description()

    # Save output of the network at the end of each epoch. The Generator
    utils.saveModels(G, gen_optimizer, epoch_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, dis_optimizer, epoch_count,
                     checkpoint_pattern % ('D', epoch_count))

    tqdm.write('[!] Model Saved!')

    return np.nanmean(total_loss_pix),\
        np.nanmean(total_loss_id), \
        np.nanmean(total_loss_G),\
        np.nanmean(total_loss_G_only),\
        np.nanmean(total_loss_D),\
        np.nanmean(total_loss_D_fake),\
        np.nanmean(total_loss_D_true),\

# Print out the experiment configurations. You can also save these to a file if
# you want them to be persistent.
print('[*] Beginning Training:')
print('\tMax Epoch: ', max_epochs)
print('\tLogging iter: ', displayIter)
print('\tSaving frequency (per epoch): ', saveIter)
print('\tModels Dumped at: ', check_point_loc)
print('\tVisuals Dumped at: ', visuals_loc)
print('\tExperiment Name: ', experiment_name)

iter_count = 0
for i in range(max_epochs):
    L = Train(gen, dis, i,0, trainLogger)
    print(L)

trainLogger.close()
