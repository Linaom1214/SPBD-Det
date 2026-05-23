import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .modules import Encoder, Encoder2


class ViG2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Encoder2(3, 32)
        self.layer2 = Encoder2(32, 64)
        self.layer3 = Encoder2(64, 128)
        self.layer4 = Encoder2(128, 256)

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)

    def forward(self, input):
        _, _, h, w = input.size()
        e1 = self.layer1(input) # 16,256,256
        e2 = self.layer2(e1)  # 32,128,128
        e3 = self.layer3(e2) #  64,64,64
        e4 = self.layer4(e3) # 128 
        
        return e1, e2, e3, e4 
    
    def switch_to_deploy(self):
        for module in self.children():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

class ViG(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Encoder(3, 32)
        self.layer2 = Encoder(32, 64)
        self.layer3 = Encoder(64, 128)
        self.layer4 = Encoder(128, 256)

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)

    def forward(self, input):
        _, _, h, w = input.size()
        e1 = self.layer1(input) # 16,256,256
        e2 = self.layer2(e1)  # 32,128,128
        e3 = self.layer3(e2) #  64,64,64
        e4 = self.layer4(e3) # 128 
        
        return e1, e2, e3, e4 
    
    def switch_to_deploy(self):
        for module in self.children():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = tiny_vit_21m_512(pretrained=True)
        self.freeze_layers(self.model, ['adapter'])
     
    
    def forward(self, input):
        e1, e2, e3 = self.model(input)
        # print(e1.size(), e2.size(), e3.size())
        return e1, e2, e3

    '''
    torch.Size([1, 192, 64, 64])
    torch.Size([1, 384, 32, 32])
    torch.Size([1, 576, 16, 16])
    '''
    def freeze_layers(self, model, layers_to_freeze):
        """
        冻结模型中的指定层。

        Args:
        model: 要冻结层的模型。
        layers_to_freeze: 要冻结的层名称列表。
        """
        for name, param in model.named_parameters():
            if not any(layer in name for layer in layers_to_freeze):  # 只训练 Adapter 层
                param.requires_grad = False

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder: nn.Module = timm.create_model(
            model_name="resnet50d", features_only=True, out_indices=range(1, 4)
        )
    
    def forward(self, x):
        return self.encoder(x)

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet_Feat(3)
    
    def forward(self, x):
        e1, e2, e3 = self.resnet(x)
        return e1, e2, e3


if __name__ == "__main__":
    model = ViG2()
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    for o in y:
        print(o.size())
