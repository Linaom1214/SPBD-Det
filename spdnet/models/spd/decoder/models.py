import torch
import torch.nn as nn

from typing import Tuple
from .modules import MLP, LayerNorm2d, PositionEmbeddingRandom, Identity
from .transformer import TwoWayTransformer

import torch.nn.functional as F

class SobelMaskDecoder(nn.Module):
    def __init__(self, out_dim):
        super(SobelMaskDecoder, self).__init__()
        mask_in_chans = 16
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, out_dim, kernel_size=1),
        )
    def forward(self, image):
        return self.mask_downscaling(image)

class MultiQueryMaskDecoder(nn.Module):
    def __init__(self, num_masks = 2, 
                    encoder_out_features = [64, 128, 256],
                    mask_depths = [1, 2, 3], 
                    mask_num_head = 8,
                    mask_mlp_dim = 2048,
                    step_size=0.1, 
                    t_total=1.0,
                    alpha=0.5, 
                    beta=0.5,
                    sobel=True,
                    ) -> None:
        """
        Modified from https://github.com/bowang-lab/MedSAM/blob/69b5185e75aaea8d175f164da67dad0442560521/segment_anything/modeling/mask_decoder.py
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.num_mask_tokens = num_masks
        self.indices = list(range(len(encoder_out_features)))[::-1]
        for block_index in self.indices:
            setattr(
                self,
                f"token_s{block_index}",
                nn.Embedding(
                    2, encoder_out_features[block_index]
                ))
            setattr(
                self,
                f"transformer_s{block_index}",
                TwoWayTransformer(
                    depth=mask_depths[block_index],
                    embedding_dim=encoder_out_features[block_index],
                    mlp_dim=mask_mlp_dim,
                    num_heads=mask_num_head,
                ),
            )
            out_dim = (
                encoder_out_features[block_index - 1]
                if block_index > 0
                else encoder_out_features[block_index]
            )
            setattr(
                self,
                f"pe_layer_s{block_index}",
                PositionEmbeddingRandom(encoder_out_features[block_index] // 2),
            )
            if block_index > 0:
                setattr(
                    self,
                    f"output_upscaling_s{block_index}",
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            encoder_out_features[block_index],
                            encoder_out_features[block_index - 1],
                            kernel_size=2,
                            stride=2,
                        ),
                        LayerNorm2d(encoder_out_features[block_index - 1]),
                        nn.GELU(),
                    ),
                )
            else:
                setattr(
                    self,
                    f"output_upscaling_s{block_index}",
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            encoder_out_features[block_index],
                            encoder_out_features[block_index] // 2,
                            kernel_size=2,
                            stride=2,
                        ),
                        LayerNorm2d(encoder_out_features[block_index] // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            encoder_out_features[block_index] // 2,
                            encoder_out_features[block_index],
                            kernel_size=2,
                            stride=2,
                        ),
                        LayerNorm2d(encoder_out_features[block_index]),
                        nn.GELU(),
                    ),
                )

        self.out = nn.ModuleList(
                    [
                        MLP(
                            encoder_out_features[block_index],
                            encoder_out_features[block_index],
                            out_dim,
                            3,
                        )
                        for _ in range(self.num_mask_tokens)
                    ]
                )
        self.pixel_embed = Identity()
        self.sobel = EBS_Module(in_channels=1, alpha=alpha, beta=beta, step_size=step_size, t_total=t_total)
        self.sobel_decoder = SobelMaskDecoder(encoder_out_features[-1])
        self.use_sobel = sobel
        if not sobel:
            self.sobel = Identity()
            self.sobel_decoder = Identity()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        org_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder

        Returns:
          torch.Tensor: batched predicted masks
        """
        high_feat = image_embeddings[0]
        image_embeddings = image_embeddings[1:]
        if org_image.size(1) == 3:
            org_image = org_image[:, 0, ...].unsqueeze(1)
        img_token = None
        if self.use_sobel:
            img = self.sobel(org_image)
            img_token = self.sobel_decoder(img)
        previous_feat = torch.zeros_like(image_embeddings[-1])
        current_feat = torch.zeros_like(image_embeddings[-1])
        for block_index in self.indices:
            query_tokens = getattr(self, f"token_s{block_index}").weight.unsqueeze(0).expand(
                image_embeddings[block_index].size(0), -1, -1
            )
            feat = image_embeddings[block_index]
            # Expand per-image data in batch direction to be per-mask
            if feat.shape[0] != query_tokens.shape[0]:
                current_feat = torch.repeat_interleave(
                    feat, query_tokens.shape[0], dim=0
                )
            else:
                current_feat = feat
            current_feat = current_feat + previous_feat

            b, c, h, w = current_feat.shape

            pos_src = torch.repeat_interleave(
                getattr(self, f"pe_layer_s{block_index}")(feat.shape[2:])
                .unsqueeze(0)
                .cpu(),
                query_tokens.shape[0],
                dim=0,
            ).to(current_feat.device)

            if block_index == self.indices[0] and img_token is not None:
                img_token = F.interpolate(img_token, size=(h, w), mode="bilinear", align_corners=False)
                current_feat = current_feat + img_token
            att_tokens, current_feat = getattr(self, f"transformer_s{block_index}")(
                current_feat, pos_src, query_tokens
            )# 图像特征、位置编码、查询编码
            current_feat = current_feat.transpose(1, 2).view(b, c, h, w)
            if block_index > 0:
                current_feat = previous_feat = getattr(
                    self, f"output_upscaling_s{block_index}"
                )(current_feat)
            else:
                dc1, ln1, act1, dc2, ln2, act2 = getattr(self, f"output_upscaling_s{block_index}")
                current_feat = act1(ln1(dc1(current_feat))) + high_feat
                current_feat = act2(ln2(dc2(current_feat)))

        hy = []
        for i in range(self.num_mask_tokens):
            hy.append(self.out[i](att_tokens[:, i, :]))
        final_query_tokens = torch.stack(hy,dim=1)

        b, c, h, w = current_feat.shape
        masks = (final_query_tokens @ current_feat.view(b, c, h * w)).view(b, -1, h, w)
        return masks, None


class EBS_Module(nn.Module):
    def __init__(self, in_channels=1, alpha=0.5, beta=0.5, step_size=0.1, t_total=1.0):
        super().__init__()
        # 可微分算子
        self.grad_term = LearnableSobelTerm(in_channels=in_channels)  # 对应α·∇_Sobel I项
        self.noise_term = StochasticNoiseTerm(in_channels=in_channels)  # 对应β·f_MLP(N)项
    
        # 可学习系数（建议初始化为论文中的α=β=0.5）
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
    
        # 时间积分器
        self.integrator = EulerIntegrator(step_size=step_size, t_total=t_total)

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        w = self.grad_term.base_edge(x)

        def sde_func(w):
            G = self.grad_term(x)
            F = self.noise_term(x, w)
            return self.alpha * G - self.beta * F

        return self.integrator(sde_func, w)

class LearnableSobelTerm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_x = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.conv_y = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)

        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -1, -1],
                                       [0, 0, 0],
                                       [1, 1, 1]], dtype=torch.float32)
        self.conv_x.weight.data = sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.conv_y.weight.data = sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
    
        # 动态幅度调制
        self.amp = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, 3, padding=1) 
        )

    def base_edge(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        return torch.sqrt(grad_x**2 + grad_y**2)

    def forward(self, x):
        grad = self.base_edge(x)
        return grad * torch.sigmoid(self.amp(x))

class StochasticNoiseTerm(nn.Module):
    def __init__(self, in_channels, latent_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + 1, latent_dim, 1),
            nn.LeakyReLU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.InstanceNorm2d(latent_dim),
            nn.Conv2d(latent_dim, in_channels, 1)
        )

    def forward(self, x, w):
        x_cond = torch.cat([w, x], dim=1)
        return self.mlp(x_cond)

class EulerIntegrator(nn.Module):
    def __init__(self, step_size=0.1, t_total=1.0):
        super().__init__()
        self.step_size = step_size
        self.t_total = t_total

    def forward(self, func, y0):
        y = y0
        for step in range(int(self.t_total / self.step_size)):
            y = y.detach()
            y.requires_grad_(True)
            y = y + self.step_size * func(y)
            if (step + 1) % 5 == 0:
                y = y.detach().requires_grad_(True)
        return y