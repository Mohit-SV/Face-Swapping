import torch
from torch.utils.data import Dataset
import numpy as np  
import cv2
import os
from img_utils import rgb2tensor 
import torchvision.transforms as T
import natsort
from img_utils import rgb2tensor  
from img_utils import bgr2tensor
from img_utils import tensor2rgb 

# Dataset used making input dict containing source, target, swap, transfer, 
# groundtruth and mask
class SwappedDatasetLoader(Dataset):

    def __init__(self, data_file, small_root, train, transform=None, resize=256):
        self.prefix = None
        self.resize = 256
        self.main_dir = data_file
        self.small_root = small_root
        
        if train == True:
            files = os.scandir(self.main_dir)
        else:
            files = os.scandir(self.small_root)
            self.main_dir = self.small_root
            
        all = [f.name for f in files]
        all = natsort.natsorted(all)

        new_dataset = []
        for i in range(0, len(all), 4):
            for item in all[i:i+4]:
                new_dataset.append(all[i:i+4])

        self.data = new_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            target = cv2.imread(self.main_dir+str(self.data[index][0]))
            source = cv2.imread(self.main_dir+str(self.data[index][1]))
            mask = cv2.imread(self.main_dir+str(self.data[index][2]))
            swap = cv2.imread(self.main_dir+str(self.data[index][3]))
        except:
            # wrong input take other input
            print("wrong input data file!")
            target = cv2.imread(self.main_dir+"0000_bg_9999.png")
            source = cv2.imread(self.main_dir+"0000_fg_0.png")
            mask = cv2.imread(self.main_dir+"0000_mask_9999_0.png")
            swap = cv2.imread(self.main_dir+"0000_sw_9999_0.png")
        
        src = tensor2rgb(rgb2tensor(swap))
        dst = tensor2rgb(rgb2tensor(target))
        src_mask = mask
       
        blending = 'pyr'
        
        # poisson blending
        if blending == 'poisson':

            max_x = 0
            min_x = 1000000 
            max_y = 0
            min_y = 1000000
            for j in range(src.shape[0]):
                for d in range(src.shape[1]):
                    if src[j][d][0] != 0 and src[j][d][1]!=0 and src[j][d][2]!=0:
                        if j > max_x:
                            max_x = j
                             
                        if j < min_x:
                            min_x = j
                            
                        if d > max_y:
                            max_y = d
                            
                        if d < min_y:
                            min_y = d
                            
            a = int((min_y + max_y)/2)
            b = int((min_x + max_x)/2)
            gr = cv2.seamlessClone(src, dst, src_mask, (a,b), cv2.NORMAL_CLONE)

        new_mask = np.zeros(shape=mask.shape)
        for item in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[item][j][0] != 0 or mask[item][1][1] != 0 or mask[item][2][2] !=0:
                    new_mask[item][j] = [1,1,1]

        # make transferred input for input to generator
        transfer = swap * new_mask + target * (np.ones(shape=new_mask.shape) - new_mask)
        
        # alpha blending
        if blending == 'alpha':
            
            # other variant of alpha blending with overlay
            A = swap + (target*(1-mask/255))
            alpha = 0.8
            gr = cv2.addWeighted(np.uint8(A), alpha, np.uint8(target), 1-alpha, 0.5)
            
        # laplacian pyramid blending for 6 pyramids
        if blending == 'pyr':
            def GP(img):
                G = img.copy()
                gp = [G]
                for i in range(6):
                    G = cv2.pyrDown(G)
                    gp.append(np.float32(G))
                return gp

            def LP(gp):
                lp = [gp[5]]
                for i in range(5,0,-1):
                    GE = cv2.pyrUp(gp[i])
                    L = np.subtract(gp[i-1],GE)
                    lp.append(L)
                return lp
             
            # make gaussian pyrs for mask
            mask_p = GP(mask/255)
            
            # make gaus and lapl pyr for source and target 
            A_l = LP(GP(swap))
            B_l = LP(GP(target))  
            
            # blend
            L = []
            count = 5
            for a, b in zip(A_l, B_l):
                l = a * mask_p[count] + b * (1 - mask_p[count])
                L.append(l)
                count = count -1
        
            # convert to right format otherwise an error will be thrown
            total = L[0]
            for i in range(1,6):
                total = cv2.pyrUp(total)
                total = cv2.add(total, np.float32(L[i]))
       
            gr = total

        # resize images to 256x256, convert to tensors and return dict
        transform = T.Resize(256)
        image_dict = {'source': transform(rgb2tensor(source)).float(),
        'target': transform(rgb2tensor(target)).float(),
        'swap' :transform(rgb2tensor(swap)).float(),
        'mask' : transform(rgb2tensor(mask)).float(),
        'gr': transform(rgb2tensor(gr)).float(),
        'transfer': transform(rgb2tensor(transfer)).float()
        }

        return image_dict

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    # It is always a good practice to have separate debug section for your
    # functions. Test if your dataloader is working here. This template creates
    # an instance of your dataloader and loads 20 instances from the dataset.
    # Fill in the missing part. This section is only run when the current file
    # is run and ignored when this file is imported.

    # This points to the root of the dataset
    data_root = '/content/drive/MyDrive/Colab Notebooks/cv2_2022_assignment3/gan_blender_release/dataset/small_dataset'
    
    # This points to a file that contains the list of the filenames to be
    # loaded.
    print('[+] Init dataloader')
    # Fill in your dataset initializations
    testSet = SwappedDatasetLoader(data_root)#, prefix=None)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=1,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(20):
        a = time.time()
        i, (images) = next(enu)
        b = time.time()
        # Uncomment to use a prettily printed version of a dict returned by the
        # dataloader.
        # printTensorList(images[0], True)
        print('[*] Time taken: ', b - a)