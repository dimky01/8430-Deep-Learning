import torch
from torch import nn
from torchvision.models import inception_v3
import cv2
import multiprocessing
import numpy as np
import glob
import os
import warnings
from scipy import linalg
import matplotlib.pyplot as plt
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.InceptionNet = inception_v3(pretrained=True)
        self.InceptionNet.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.InceptionNet(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations


# In[6]:


def getActivations(images, batchSize):
    
    assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +                                              ", but got {}".format(images.shape)

    noofImg = images.shape[0]
    inceptionNet = InceptionNetwork()
    inceptionNet = inceptionNet.to(device)
    inceptionNet.eval()
    n_batches = int(np.ceil(noofImg  / batchSize))
    inceptionAct = np.zeros((noofImg, 2048), dtype=np.float32)
    for batch_idx in range(batchSize):
        start_idx = batchSize * batch_idx
        end_idx = batchSize * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = ims.to(device)
        activations = inceptionNet(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 2048), activations.shape)
        inceptionAct[start_idx:end_idx, :] = activations
    return inceptionAct


# In[7]:


def activationStatistics(images, batchSize):
    
    act = getActivations(images, batchSize)
    mean = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mean, sigma


# In[9]:


def calFrechetDist(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
#             raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# In[8]:


def ImgPreprocessing(img):
    
    assert img.shape[2] == 3
    assert len(img.shape) == 3
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    img = cv2.resize(img, (299, 299))
    img = np.rollaxis(img, axis=2)
    img = torch.from_numpy(img)
    assert img.max() <= 1.0
    assert img.min() >= 0.0
    assert img.dtype == torch.float32
    assert img.shape == (3, 299, 299)

    return img


def multiImgPreprocess(images, use_multiprocessing):
    
    if use_multiprocessing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            jobs = []
            for img in images:
                job = pool.apply_async(ImgPreprocessing, (img,))
                jobs.append(job)
            final_images = torch.zeros(images.shape[0], 3, 299, 299)
            for idx, job in enumerate(jobs):
                img = job.get()
                final_images[idx] = img#job.get()
    else:
        final_images = torch.stack([ImgPreprocessing(img) for img in images], dim=0)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float32
    return final_images


# In[10]:


def calFid(images1, images2, use_multiprocessing, batchSize):
    
    images1 = multiImgPreprocess(images1, use_multiprocessing)
    images2 = multiImgPreprocess(images2, use_multiprocessing)
    mean1, sigma1 = activationStatistics(images1, batchSize)
    mean2, sigma2 = activationStatistics(images2, batchSize)
    fid = calFrechetDist(mean1, sigma1,mean2, sigma2)
    return fid


# In[11]:


def loadImages(path, iteration):
    
    image_paths = []
    image_extensions = ["png"]
    for ext in image_extensions:
        #print("Looking for images in", os.path.join(path, "*.{}".format(ext)))
        for impath in glob.glob(os.path.join(path, "*.{}".format(ext))):
            image_paths.append(impath)
    #sort the images by name
    image_paths = sorted(image_paths)
    image_paths.sort(key = len)
    #compare only the most recently generated fake and real image
    image_paths = image_paths[0:iteration]
    first_image = cv2.imread(image_paths[0])

    H, W = first_image.shape[:2]
    image_paths.sort()
    image_paths = image_paths
    final_images = np.zeros((len(image_paths), H, W, 3), dtype=first_image.dtype)
    for idx, impath in enumerate(image_paths):
        img = cv2.imread(impath)
        img = img[:, :, ::-1] # Convert from BGR to RGB
        assert img.dtype == final_images.dtype
        final_images[idx] = img
    return final_images



def fidScore(path1,path2, iteration, batchSize=16): #, acgan=False):
    
    images1 = loadImages(path1, iteration) #, acgan)
    images2 = loadImages(path2, iteration) #, acgan)
    FID = calFid(images1, images2, False, batchSize)
#     print('FID VALUE:',FID)
    return FID