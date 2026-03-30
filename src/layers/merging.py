def merge_patch_to_context(x_context, pred_patch):
    B, C, H, W = x_context.shape
    mask_h, mask_w = pred_patch.shape[2], pred_patch.shape[3]
    y1 = (H - mask_h) // 2
    x1 = (W - mask_w) // 2
    x_recon = x_context.clone()
    x_recon[:, :, y1:y1+mask_h, x1:x1+mask_w] = pred_patch
    return x_recon
