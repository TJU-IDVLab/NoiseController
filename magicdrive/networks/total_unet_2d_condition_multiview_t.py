import sys
from ..networks.unet_2d_condition_multiview_t import UNet2DConditionModelMultiviewT

class Total_UNet_BI(UNet2DConditionModelMultiviewT):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


if __name__ == "__main__":
    model = Total_UNet_BI()
