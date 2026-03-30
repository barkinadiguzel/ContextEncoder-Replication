def extract_center_patch(x, mask_height=64, mask_width=64):
    B, C, H, W = x.shape
    y1 = (H - mask_height) // 2
    x1 = (W - mask_width) // 2
    return x[:, :, y1:y1+mask_height, x1:x1+mask_width]
