import torch
import numpy as np
from torch.autograd import Variable


class WeightedEuclideanLossLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, crop, flux, dilmask):
        weightPos = np.zeros_like(crop.cpu().data, dtype=np.float32)
        weightNeg = np.zeros_like(crop.cpu().data, dtype=np.float32)

        distL1 = crop.data - flux.data
        distL2 = distL1 ** 2
        # the amount of positive and negative pixels
        regionPos = (dilmask.data > 0)
        regionNeg = (dilmask.data == 0)
        sumPos = regionPos.sum()
        sumNeg = regionNeg.sum()
        # balanced weight for positive and negative pixels
        weightPos[0][0] = sumNeg.float() / float(sumPos + sumNeg) * regionPos.cpu().float()
        weightPos[0][1] = sumNeg.float() / float(sumPos + sumNeg) * regionPos.cpu().float()
        weightNeg[0][0] = sumPos.float() / float(sumPos + sumNeg) * regionNeg.cpu().float()
        weightNeg[0][1] = sumPos.float() / float(sumPos + sumNeg) * regionNeg.cpu().float()
        # total loss
        loss = (distL2 * torch.from_numpy(weightPos + weightNeg).cuda()).sum() / len(crop) / 2. / (
                    weightPos + weightNeg).sum()
        ctx.save_for_backward(crop, flux, dilmask)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        crop, flux, dilmask = ctx.saved_variables
        weightPos = np.zeros_like(crop.cpu().data, dtype=np.float32)
        weightNeg = np.zeros_like(crop.cpu().data, dtype=np.float32)
        distL1 = crop.data - flux.data
        # the amount of positive and negative pixels
        regionPos = (dilmask.data > 0)
        regionNeg = (dilmask.data == 0)
        sumPos = regionPos.sum()
        sumNeg = regionNeg.sum()
        # balanced weight for positive and negative pixels
        weightPos[0][0] = sumNeg.float() / float(sumPos + sumNeg) * regionPos.cpu().float()
        weightPos[0][1] = sumNeg.float() / float(sumPos + sumNeg) * regionPos.cpu().float()
        weightNeg[0][0] = sumPos.float() / float(sumPos + sumNeg) * regionNeg.cpu().float()
        weightNeg[0][1] = sumPos.float() / float(sumPos + sumNeg) * regionNeg.cpu().float()
        grad1 = distL1 * torch.from_numpy(weightPos + weightNeg).cuda() / len(crop)
        return grad1 * grad_output, None, None


wel = WeightedEuclideanLossLayer.apply

