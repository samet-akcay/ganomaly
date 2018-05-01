##
# LIBRARIES
from __future__ import print_function

from options.train_options import TrainOptions
from lib.data.dataloader import load_data
from lib.models.models import load_model
from lib.loss.losses import l2_loss

import torch

from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot([0, 1], [1, 0], color='navy',lw=1,  linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()
        
    return roc_auc, eer

##
def demo(img_path, model):
    def pil_loader(path):
        from PIL import Image
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def imshow(org, gen, score):
        """Imshow for Tensor."""
        def unnormalize(inp):
            inp = inp.data.cpu().squeeze_(0).numpy().transpose((1, 2, 0))
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            return inp
        
        org = unnormalize(org)
        gen = unnormalize(gen)
        score = score.data.cpu().numpy()[0][0][0]
        fig = plt.figure()

        f = fig.add_subplot(1, 2, 1)
        plt.imshow(org)
        plt.axis('off')
        f.set_title('Original')
        f = fig.add_subplot(1, 2, 2)
        plt.imshow(gen)
        plt.axis('off')
        f.set_title('Generated')

        plt.suptitle("Score: {:.4f}".format(score))
        plt.pause(0.001)

    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.Scale(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])


    from torch.autograd import Variable
    img = pil_loader(img_path)
    img = Variable(transform(img)).unsqueeze(0).cuda()

    gen, zi, zo = model.netg(img)
    error = torch.mean(torch.pow((zi-zo), 2), dim=1)

    imshow(img, gen, error)

##
# ARGUMENTS
opt = TrainOptions().parse()

# # LOAD DATA
dataloader = load_data(opt)
# f = load_data(opt, 'folder')['test']
# t = load_data(opt, 'txt')['test']
# dataloader = f

# images = dataloader.dataset.imgs
# images1 = [(os.path.split(i[0])[0], os.path.split(i[0])[1], i[1], i[2]) for i in images]
# images2 = sorted(images1, key=lambda i: i[1])
# images3 = [(os.path.join(i[0], i[1]), i[2], i[3]) for i in images2]
# fnames = np.array([i[0] for i in images]).reshape(len(images),1)


# dataloader.dataset.imgs = t.dataset.imgs\

# # LOAD MODEL
model = load_model(opt, dataloader)

# # TEST MODEL
res = model.test()
print(res['auc'], res['eer'])
# labels, scores = model.test()
# labels, scores = labels.cpu().numpy(), scores.cpu().numpy()

# label = labels.astype(str)
# score = scores.astype(str)

# np.savetxt('score11.txt', np.concatenate((fnames, label, score), axis=-1), fmt="%s")

# auc, eer = roc(labels, scores)
# print(auc)


# output = torch.cat((labels.float().cuda(), scores), dim=1)
# patches = torch.chunk(output, chunks=7200)

# fs = ()
# for p in patches:
#     # ps = torch.cat((torch.max(p, dim=0)[0].view(1, 2),) * 16)
#     ps = torch.max(p, dim=0)[0].view(1, 2)
#     fs = fs + (ps, )
# fs = torch.cat(fs)

# fpr, tpr, eer = roc(fs[:, 0].cpu().numpy(), fs[:, 1].cpu().numpy())
# fpr, tpr, eer = roc(labels.cpu().numpy(), scores.cpu().numpy())


# fpr, tpr, eer = roc(labels, scores)


# ## LOAD WEIGHTS
# # Load the weights of netg and nete.
# path = "./output/{}.{}.{}.{}.{}/train/weights/netG.pth".format(
#     opt.dataset,
#     model.name().lower(),
#     model.opt.isize,
#     model.opt.niter,
#     int(model.opt.alpha)
# )
# pretrained_dict = torch.load(path)['state_dict']

# try:
#     model.netg.load_state_dict(pretrained_dict)
# except IOError:
#     raise IOError("netG weights not found")

# model.netg.eval()



# demo(dataloader.dataset.imgs[0][0], model)
