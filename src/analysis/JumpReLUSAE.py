import torch.nn as nn
import torch

class JumpReLUSAE(nn.Module):
  def __init__(self, d_in, d_sae, affine_skip_connection=False):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
    self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_in))
    if affine_skip_connection:
      self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
    else:
      self.affine_skip_connection = None

  def encode(self, input_acts):
    pre_acts = input_acts @ self.w_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts, pre_acts

  def decode(self, acts):
    return acts @ self.w_dec + self.b_dec

  def forward(self, x):
    acts = self.encode(x)
    recon = self.decode(acts)
    if self.affine_skip_connection is not None:
      return recon + x @ self.affine_skip_connection
    return recon