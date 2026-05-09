import os
import requests
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()


class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)


class JigsawHead(nn.Module):
    """Jigswa + linear + l2norm"""
    def __init__(self, dim_in, dim_out, k=9, head='linear'):
        super(JigsawHead, self).__init__()

        if head == 'linear':
            self.fc1 = nn.Linear(dim_in, dim_out)
        elif head == 'mlp':
            self.fc1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_out),
            )
        else:
            raise NotImplementedError('JigSaw head not supported: {}'.format(head))
        self.fc2 = nn.Linear(dim_out * k, dim_out)
        self.l2norm = Normalize(2)
        self.k = k

    def forward(self, x):
        bsz = x.shape[0]
        x = self.fc1(x)
        # ==== shuffle ====
        # this step can be moved to data processing step
        shuffle_ids = self.get_shuffle_ids(bsz)
        x = x[shuffle_ids]
        # ==== shuffle ====
        n_img = int(bsz / self.k)
        x = x.view(n_img, -1)
        x = self.fc2(x)
        x = self.l2norm(x)
        return x

    def get_shuffle_ids(self, bsz):
        n_img = int(bsz / self.k)
        rnd_ids = [torch.randperm(self.k) for i in range(n_img)]
        rnd_ids = torch.cat(rnd_ids, dim=0)
        base_ids = torch.arange(bsz)
        base_ids = torch.div(base_ids, self.k).long()
        base_ids = base_ids * self.k
        shuffle_ids = rnd_ids + base_ids
        return shuffle_ids
