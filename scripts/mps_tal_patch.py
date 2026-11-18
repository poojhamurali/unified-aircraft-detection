"""Work around a boolean-mask indexing bug in the MPS backend.

Ultralytics' TaskAlignedAssigner.get_box_metrics both gathers and scatters
through the same boolean mask:

    bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_labels][mask_gt]
    overlaps[mask_gt]    = iou(gt_bboxes[mask_gt], pd_bboxes[mask_gt])

On Metal the gather and the scatter can disagree about how many elements the
mask selects, and training dies with

    RuntimeError: shape mismatch: value tensor of shape [3452]
    cannot be broadcast to indexing result of shape [1983]

The two counts differ run to run, so this is a correctness bug in the backend
rather than a size limit: batch 16 died in the first epoch, batch 4 survived one
epoch and died in the second. Shrinking the batch only changes how long it takes
to hit.

The replacement below computes the same two tensors by multiplying with the mask
instead of indexing by it. Both destinations start as zeros and the mask only
ever selects which entries stay non-zero, so `x * mask` and `dst[mask] = x[mask]`
produce identical results -- but multiplication never touches the broken path.

The cost is that IoU is now evaluated at every (ground truth, anchor) pair
rather than only the masked ones. At batch 4 that is a few hundred MB of
short-lived intermediates, which is why the optical runs stay at batch 4.

HRPlanesv2 provokes this and the infrared stream did not because these scenes
carry up to 127 aircraft each, and mosaic tiles four scenes into one sample.
"""

from __future__ import annotations

import torch


def _get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    na = pd_bboxes.shape[-2]
    mask = mask_gt.to(pd_scores.dtype)  # b, max_num_obj, h*w

    batch_ind = torch.arange(self.bs, device=pd_scores.device)[:, None]  # b, 1
    bbox_scores = pd_scores[batch_ind, :, gt_labels.squeeze(-1).long()] * mask

    pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)
    gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)
    overlaps = self.iou_calculation(gt_boxes, pd_boxes) * mask

    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps


def apply(device: str) -> bool:
    """Install the workaround when training on Metal. Returns whether it applied."""
    if device != "mps":
        return False
    from ultralytics.utils.tal import TaskAlignedAssigner

    TaskAlignedAssigner.get_box_metrics = _get_box_metrics
    return True
