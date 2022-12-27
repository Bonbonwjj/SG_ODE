import torch
import torch.nn.functional as F
import torchvision.transforms as T
import PIL


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
  return T.Compose(transforms)


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


def gradient_penalty(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty


def compute_losses(args, boxes_pred, boxes_gt, frame_pred, frame_real):
    loss_bbox = F.mse_loss(boxes_pred, boxes_gt) * args.bbox_loss_weight

    loss_img = F.l1_loss(frame_pred, frame_real) * args.img_loss_weight

    person_box = boxes_gt[0]
    x1 = int(person_box[0] * 64)
    y1 = int(person_box[1] * 64)
    x2 = int(person_box[2] * 64)
    y2 = int(person_box[3] * 64)
    loss_pb = F.l1_loss(frame_pred[:, :, x1:x2, y1:y2], frame_real[:, :, x1:x2, y1:y2]) * args.pb_loss_weight

    total_loss = loss_bbox + loss_img + loss_pb

    return total_loss