import torch
import torchvision.transforms as T
import PIL


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
#pytorch official

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

def imagenet_preprocess():
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

class Resize(object):
  def __init__(self, size, interp=PIL.Image.BILINEAR):
    if isinstance(size, tuple):
      H, W = size
      self.size = (W, H)
    else:
      self.size = (size, size)
    self.interp = interp

  def __call__(self, img):
    return img.resize(self.size, self.interp)

def imagenet_preprocess():
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
  lo, hi = x.min(), x.max()
  return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
  transforms = [
    T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
    T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(rescale)
  return T.Compose(transforms) #Compose()类会将transforms列表里面的transform操作进行遍历


def imagenet_deprocess_batch(imgs, rescale=True):

  if isinstance(imgs, torch.autograd.Variable):
    imgs = imgs.data
  imgs = imgs.cpu().clone()
  deprocess_fn = imagenet_deprocess(rescale_image=rescale)
  imgs_de = []
  for i in range(imgs.size(0)):
    img_de = deprocess_fn(imgs[i])[None]
    img_de = img_de.mul(255).clamp(0, 255).byte()
    imgs_de.append(img_de)
  imgs_de = torch.cat(imgs_de, dim=0)
  return imgs_de

def gradient_penalty(netD, real, fake, device="cpu"):
  BATCH_SIZE, C, H, W = real.shape  # (1,3,64,64)
  alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)  # (1,1,1,1)→(1,3,64,64)
  interpolated_images = real * alpha + fake * (1 - alpha)  # fake(1,3,64,64)

  # Calculate critic scores
  mixed_scores = netD(interpolated_images)

  # Take the gradient of the scores with respect to the images
  gradient = torch.autograd.grad(
    inputs=interpolated_images,
    outputs=mixed_scores,
    grad_outputs=torch.ones_like(mixed_scores),
    create_graph=True,
    retain_graph=True,
  )[0]
  gradient = gradient.view(gradient.shape[0], -1)  # flatten(64,4096)
  gradient_norm = gradient.norm(2, dim=1)  # L2
  gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
  return gradient_penalty