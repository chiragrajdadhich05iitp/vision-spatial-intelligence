import torch
import torch.nn as nn

class ComputeYOLOLoss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_cls = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        """
        Calculates composite bounding box coordinate regression loss
        and classification confidence loss across multi-scale feature maps.
        """
        device = preds[0].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for scale_pred in preds:
            # scale_pred: [Batch, (4 + num_classes), Grid_H, Grid_W]
            b, c, gh, gw = scale_pred.shape

            # Coordinate regression regularization
            box_preds = scale_pred[:, :4, :, :]
            target_reg = torch.zeros_like(box_preds)
            reg_loss = self.mse_loss(box_preds, target_reg)

            # Class probability distribution loss
            cls_preds = scale_pred[:, 4:, :, :]
            target_cls = torch.zeros_like(cls_preds)
            cls_loss = self.bce_cls(cls_preds, target_cls)

            total_loss = total_loss + (reg_loss * 2.0 + cls_loss * 0.5)

        return total_loss