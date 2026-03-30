import torch

def reconstruction_loss(pred, target):
    return ((pred - target) ** 2).mean()

def adversarial_loss(discriminator, real, fake):
    real_loss = torch.log(discriminator(real) + 1e-8).mean()
    fake_loss = torch.log(1 - discriminator(fake) + 1e-8).mean()
    return -(real_loss + fake_loss)

def total_loss(rec_loss, adv_loss, lambda_rec=0.999, lambda_adv=0.001):
    return lambda_rec * rec_loss + lambda_adv * adv_loss
