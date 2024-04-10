import os
import random
import numpy as np
import torch

def set_seed(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('Set random seed as {} for pytorch'.format(seed))


def standardize_label(factor_input):
    factor_output = (factor_input - np.nanmean(factor_input)) / np.nanstd(factor_input)
    return factor_output


def pearson_r_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    x_square_sum = torch.sum(xm * xm)
    y_square_sum = torch.sum(ym * ym)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return -torch.mean(r)


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    x_square_sum = torch.sum(xm * xm)
    y_square_sum = torch.sum(ym * ym)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return torch.mean(r)