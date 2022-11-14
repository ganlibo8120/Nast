import torch
import torch.nn as nn
import torch.nn.functional as F


def quality_focal_loss(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert (
        len(target) == 2
    ), """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = torch.nonzero((label >= 0) & (label < bg_class_ind), as_tuple=False).squeeze(
        1
    )
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss

def extra_loss(pred, target):
    label, score = target
    n,c = pred.shape
    allC = []
    #loss = 0
    Loss = 0
    for i in range(n):
        if label[i] < c and label[i] not in allC:
            allC.append(label[i])
            idx = label == label[i]
            scorc = score[idx]
            predc = pred[:,label[i]]
            predc = predc[idx]
            Max,index = torch.max(predc)
            scorc = scorc - scorc[index]
            loss = -scorc*torch.log(predc)
            Loss += torch.mean(loss)
    return Loss

if __name__ == '__main__':
    preds = -10000*torch.ones([5,80])
    target1 = torch.randint(0,80,[5,])
    target2 = torch.rand([5,])
    preds[:,target1] = 0.5
    quality_focal_loss(preds,(target1,target2))
