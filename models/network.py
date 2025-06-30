import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class DropPath(nn.Module):
    """Stochastic Depth丢弃路径"""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class ConvX(nn.Module):
    """基础卷积块"""
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, use_act=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                             groups=groups, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = nn.GELU() if use_act else nn.Identity()

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        return out


class LearnablePool2d(nn.Module):
    """可学习的池化层"""
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(1, 1, kernel_size, kernel_size), requires_grad=True)
        nn.init.normal_(self.weight, 0, 0.01)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        weight = self.weight.repeat(self.dim, 1, 1, 1)
        out = nn.functional.conv2d(x, weight, None, self.stride, self.padding, groups=self.dim)
        return self.norm(out)


class ChannelLearnablePool2d(nn.Module):
    """通道可学习池化层"""
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, 
                             groups=dim, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.conv(x)
        return self.norm(out)


class PyramidFC(nn.Module):
    """金字塔特征聚合模块"""
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, use_dw=False):
        super(PyramidFC, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d

        self.branch_1 = nn.Sequential(
            block(inplanes, kernel_size=3, stride=1, padding=1),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_2 = nn.Sequential(
            block(inplanes, kernel_size=5, stride=2, padding=2),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_3 = nn.Sequential(
            block(inplanes, kernel_size=7, stride=3, padding=3),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.branch_1(x)
        x2 = F.interpolate(self.branch_2(x), size=(h, w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.branch_3(x), size=(h, w), mode='bilinear', align_corners=False)
        x4 = self.branch_4(x)
        out = self.act(x1 + x2 + x3 + x4)
        return out
    

class BottleNeck(nn.Module):
    """瓶颈块"""
    def __init__(self, in_planes, out_planes, stride=1, expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path=0.0):
        super(BottleNeck, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d
        expand_planes = int(in_planes*expand_ratio)
        mid_planes = int(out_planes*mlp_ratio)

        self.smlp = nn.Sequential(
            PyramidFC(in_planes, expand_planes, kernel_size=3, stride=stride, use_dw=use_dw),
            ConvX(expand_planes, in_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )
        self.cmlp = nn.Sequential(
            ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, use_act=True),
            block(mid_planes, kernel_size=3, stride=stride, padding=1) if stride==1 else ConvX(mid_planes, mid_planes, groups=mid_planes, kernel_size=3, stride=2, use_act=False),
            ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )

        self.skip = nn.Identity()
        if stride == 2 and in_planes != out_planes:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=2, use_act=False),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.smlp(x)) + x
        x = self.drop_path(self.cmlp(x)) + self.skip(x)
        return x


class AttentionModule(nn.Module):
    """注意力模块"""
    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ch_att = self.channel_att(x)
        x = x * ch_att
        
        # 空间注意力
        sp_att = self.spatial_att(x)
        x = x * sp_att
        
        return x


class FusionBlock(nn.Module):
    """融合块"""
    def __init__(self, dim):
        super(FusionBlock, self).__init__()
        self.attention1 = AttentionModule(dim)
        self.attention2 = AttentionModule(dim)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )
        
        self.weight_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, 2, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        # 应用注意力机制
        att1 = self.attention1(x1)
        att2 = self.attention2(x2)
        
        # 连接特征
        concat_feat = torch.cat([att1, att2], dim=1)
        
        # 计算融合权重
        weights = self.weight_conv(concat_feat)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        
        # 加权融合
        fused = w1 * att1 + w2 * att2
        
        # 进一步处理融合特征
        refined = self.fusion_conv(concat_feat)
        
        return fused + refined

class UpsampleBlock(nn.Module):
    """上采样块，使用双线性插值+卷积避免棋盘格效应"""
    def __init__(self, in_dim, out_dim, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        # 使用双线性插值上采样
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # 使用卷积平滑特征
        x = self.conv(x)
        return x


class MLP_MFF(nn.Module):
    """多焦点图像融合网络"""
    def __init__(self, dims=16, layers=[1, 1, 1, 1], block=BottleNeck, 
                 expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path_rate=0.):
        super(MLP_MFF, self).__init__()
        self.block = block
        self.expand_ratio = expand_ratio
        self.mlp_ratio = mlp_ratio
        self.use_dw = use_dw
        self.drop_path_rate = drop_path_rate

        if isinstance(dims, int):
            dims = [dims//2, dims, dims*2, dims*4, dims*8]
        else:
            dims = [dims[0]//2] + dims


        self.first_conv = ConvX(1, dims[0], 1, 3, 2, use_act=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        # 编码器层
        self.layer1 = self._make_layers(dims[0], dims[1], layers[0], stride=2, drop_path=dpr[:layers[0]])
        self.layer2 = self._make_layers(dims[1], dims[2], layers[1], stride=2, drop_path=dpr[layers[0]:sum(layers[:2])])
        self.layer3 = self._make_layers(dims[2], dims[3], layers[2], stride=2, drop_path=dpr[sum(layers[:2]):sum(layers[:3])])
        self.layer4 = self._make_layers(dims[3], dims[4], layers[3], stride=2, drop_path=dpr[sum(layers[:3]):sum(layers[:4])])

        # 融合模块
        self.fusion0 = FusionBlock(dims[0])  # 新增最浅层融合模块
        self.fusion1 = FusionBlock(dims[1])
        self.fusion2 = FusionBlock(dims[2])
        self.fusion3 = FusionBlock(dims[3])
        self.fusion4 = FusionBlock(dims[4])

        # 解码器层 - 使用上采样+卷积替代转置卷积
        self.decode4 = UpsampleBlock(dims[4], dims[3], scale_factor=2)
        self.decode3 = UpsampleBlock(dims[3], dims[2], scale_factor=2)
        self.decode2 = UpsampleBlock(dims[2], dims[1], scale_factor=2)
        self.decode1 = UpsampleBlock(dims[1], dims[0], scale_factor=2)

        # 最终输出层
        self.final_upsample = UpsampleBlock(dims[0], dims[0], scale_factor=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(dims[0], 1, 3, padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        self.init_params()

    def _make_layers(self, inputs, outputs, num_block, stride, drop_path):
        layers = [self.block(inputs, outputs, stride, self.expand_ratio, self.mlp_ratio, self.use_dw, drop_path[0])]

        for i in range(1, num_block):
            layers.append(self.block(outputs, outputs, 1, self.expand_ratio, self.mlp_ratio, self.use_dw, drop_path[i]))
            
        return nn.Sequential(*layers)

    def init_params(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img1, img2):
        """
        前向传播
        Args:
            img1: 第一张多焦点图像 [B, 1, H, W]
            img2: 第二张多焦点图像 [B, 1, H, W]
        Returns:
            融合后的图像 [B, 1, H, W]
        """
        # 对两张图像分别进行特征提取
        # 第一张图像的特征提取
        x1_0 = self.first_conv(img1)
        x1_1 = self.layer1(x1_0)
        x1_2 = self.layer2(x1_1)
        x1_3 = self.layer3(x1_2)
        x1_4 = self.layer4(x1_3)

        # 第二张图像的特征提取
        x2_0 = self.first_conv(img2)
        x2_1 = self.layer1(x2_0)
        x2_2 = self.layer2(x2_1)
        x2_3 = self.layer3(x2_2)
        x2_4 = self.layer4(x2_3)

        # 多尺度特征融合
        fused_4 = self.fusion4(x1_4, x2_4)
        fused_3 = self.fusion3(x1_3, x2_3)
        fused_2 = self.fusion2(x1_2, x2_2)
        fused_1 = self.fusion1(x1_1, x2_1)

        # 解码器 - 逐步上采样并融合跳跃连接
        up4 = self.decode4(fused_4)
        up4 = up4 + fused_3  # 跳跃连接
        
        up3 = self.decode3(up4)
        up3 = up3 + fused_2  # 跳跃连接
        
        up2 = self.decode2(up3)
        up2 = up2 + fused_1  # 跳跃连接
        
        up1 = self.decode1(up2)
        # 处理尺寸匹配问题
        if up1.shape[2:] != x1_0.shape[2:]:
            up1 = F.interpolate(up1, size=x1_0.shape[2:], mode='bilinear', align_corners=False)
        
        # 对x1_0和x2_0进行融合
        fused_0 = self.fusion0(x1_0, x2_0)  # 使用专门的fusion0进行融合
        up1 = up1 + fused_0  # 结合融合后的特征

        # 最终输出
        final_up = self.final_upsample(up1)
        # 融合原始分辨率的特征
        final_up = final_up + img1 + img2  # 使用加法融合原始图像特征
        output_end2end = self.final_conv(final_up)
        return output_end2end


def test_network():
    """测试网络的功能"""
    print("=== 多焦点图像融合网络测试 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = MLP_MFF().to(device)
    model.eval()
    
    # 创建示例输入
    batch_size = 1
    input1 = torch.randn(batch_size, 1, 256, 256).to(device)
    input2 = torch.randn(batch_size, 1, 256, 256).to(device)
    
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(input1, input2))
    # 将FLOPs转换为G单位，参数量转换为M单位
    flops_g = flops / 1e9
    params_m = params / 1e6
    
    print(f"FLOPs: {flops_g:.3f}G")
    print(f"模型大小: {params_m:.3f}M")
    
    # 测试前向传播
    with torch.no_grad():
        output = model(input1, input2)
        print(f"输出形状: {output.shape}")

if __name__ == "__main__":
    test_network()