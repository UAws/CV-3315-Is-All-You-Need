import torch
from torch import nn
import torch.nn.functional as F

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class BasicDecoder(BaseDecodeHead):

    def __init__(self, n_class, input_size, interpolate_mode='bilinear', **kwargs):
        super().__init__(**kwargs)
        self.interpolate_mode = interpolate_mode

        self.input_size = input_size
        self.upsample_cov1 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, output_padding=1, bias=False)  # upsample
        self.upsample_cov2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)  # upsample  -> 1/4
        self.upsample_cov3 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1,
                                                bias=False)  # upsample  -> 1/2

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.upsample_cov1(x)
        x = self.upsample_cov2(x)
        x = self.upsample_cov3(x)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=True)  # resize (1/2 -> original size)
        x = resize(
            input=x,
            size=x.shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners)

        return x
