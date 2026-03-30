import torch
import torch.nn as nn
from backbone.encoder_conv import EncoderConv
from backbone.decoder_deconv import DecoderDeconv
from backbone.discriminator_conv import DiscriminatorConv
from layers.masking import mask_center
from layers.patch_extractor import extract_center_patch
from loss.losses import reconstruction_loss, adversarial_loss, total_loss
from config import LAMBDA_REC, LAMBDA_ADV, DEVICE

class ContextEncoderPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator_encoder = EncoderConv()
        self.generator_decoder = DecoderDeconv()
        self.discriminator = DiscriminatorConv()

    def forward(self, x):
        x = x.to(DEVICE)
        x_context = mask_center(x)
        x_center = extract_center_patch(x)

        latent = self.generator_encoder(x_context)
        pred_patch = self.generator_decoder(latent)

        rec = reconstruction_loss(pred_patch, x_center)
        adv = adversarial_loss(self.discriminator, x_center, pred_patch)
        loss = total_loss(rec, adv, LAMBDA_REC, LAMBDA_ADV)

        return pred_patch, loss, rec, adv
