"""
独立 TOPIQ（Top-down Image Quality Assessment）模型实现。

该文件从 pyiqa 框架中抽离，用于项目内的鸟类摄影美学评分。
与 `topiq_model.py` 相比，ONNX 版本保留相同的主体结构，但针对导出做了
少量结构性约束，避免运行时依赖不稳定的动态图算子。

基于：
- TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment
- Chaofeng Chen et al., IEEE TIP 2024
- 原始实现：https://github.com/chaofengc/IQA-PyTorch

关键差异：
- 保持 ResNet50 + CFANet 主体结构不变
- 为 ONNX 兼容，将部分自适应池化路径改为固定步长池化
- 目标是导出稳定、推理可复现，而不是改变评分逻辑
"""

import os
import sys
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
from collections import OrderedDict

import timm
from tools.i18n import t as _t


# ImageNet 标准化参数
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


def _get_clones(module, N):
    """复制同构模块，供多层编码器/解码器堆叠复用。"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """根据字符串名称返回激活函数。"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """
    将分布预测转换为 MOS 分数。

    对于 AVA 这类具备详细评分标签的数据集，模型输出的是 1~10 分的
    概率分布，最终通过加权求和得到单个美学分数。
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score


class TransformerEncoderLayer(nn.Module):
    """自注意力编码层。"""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu',
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = src2
        src2, self.attn_map = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    """跨尺度注意力解码层。"""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu',
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, self.attn_map = self.multihead_attn(query=tgt2, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Module):
    """编码层堆叠。"""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


class TransformerDecoder(nn.Module):
    """解码层堆叠。"""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return output


class GatedConv(nn.Module):
    """门控卷积模块，用于局部质量感知特征聚合。"""

    def __init__(self, weightdim, ksz=3):
        super().__init__()

        self.splitconv = nn.Conv2d(weightdim, weightdim * 2, 1, 1, 0)
        self.act = nn.GELU()

        self.weight_blk = nn.Sequential(
            nn.Conv2d(weightdim, 64, 1, stride=1),
            nn.GELU(),
            nn.Conv2d(64, 64, ksz, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, ksz, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1, x2 = self.splitconv(x).chunk(2, dim=1)
        weight = self.weight_blk(x2)
        x1 = self.act(x1)
        return x1 * weight


class CFANet(nn.Module):
    """
    TOPIQ 的 CFANet 主体模型。

    使用 ResNet50 作为语义骨干网络，通过多尺度自注意力和跨尺度注意力，
    实现“从语义到失真”的自顶向下质量评估。
    """

    def __init__(
        self,
        semantic_model_name='resnet50',
        backbone_pretrain=False,
        use_ref=False,
        num_class=10,
        inter_dim=512,
        num_heads=4,
        num_attn_layers=1,
        dprate=0.1,
        activation='gelu',
        test_img_size=None,
    ):
        super().__init__()

        self.semantic_model_name = semantic_model_name
        self.use_ref = use_ref
        self.num_class = num_class
        self.test_img_size = test_img_size

        # =============================================================
        # ResNet50 骨干网络，仅负责提取多尺度特征
        # =============================================================
        self.semantic_model = timm.create_model(
            semantic_model_name, pretrained=backbone_pretrain, features_only=True
        )
        feature_dim_list = self.semantic_model.feature_info.channels()

        # ImageNet 归一化参数
        self.default_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

        # =============================================================
        # 自注意力与跨尺度注意力模块
        # =============================================================
        ca_layers = sa_layers = num_attn_layers
        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # 各尺度局部门控池化 + 自注意力编码
        tmp_layer = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )

        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()

        for idx, dim in enumerate(feature_dim_list):
            self.weight_pool.append(GatedConv(dim))
            self.dim_reduce.append(
                nn.Sequential(
                    nn.Conv2d(dim, inter_dim, 1, 1),
                    self.act_layer,
                )
            )
            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))

        # 跨尺度注意力：从高层语义逐步向低层细节传播
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # 最终的注意力池化和评分头
        self.attn_pool = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )

        self.score_linear = nn.Sequential(
            nn.LayerNorm(inter_dim),
            nn.Linear(inter_dim, inter_dim),
            self.act_layer,
            nn.LayerNorm(inter_dim),
            nn.Linear(inter_dim, inter_dim),
            self.act_layer,
            nn.Linear(inter_dim, self.num_class),
            nn.Softmax(dim=-1),
        )

        # 可学习位置编码
        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))

        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)
        self._init_linear(self.sa_attn_blks)
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        self.eps = 1e-8

    def _init_linear(self, m):
        """初始化线性层参数。"""
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def preprocess(self, x):
        """按 ImageNet 均值方差标准化输入。"""
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def fix_bn(self, model):
        """冻结 BatchNorm 参数，确保评估阶段统计量稳定。"""
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def forward_cross_attention(self, x):
        """
        执行主干特征提取与跨尺度注意力融合。

        ONNX 版本在这里保留与原版一致的总体数据流，但对某些动态池化路径做了
        导出友好的替换。
        """
        if not self.training and self.test_img_size is not None:
            x = TF.resize(x, self.test_img_size, antialias=True)

        x = self.preprocess(x)
        dist_feat_list = self.semantic_model(x)
        self.fix_bn(self.semantic_model)
        self.semantic_model.eval()

        start_level = 0
        end_level = len(dist_feat_list)

        pos_emb = torch.cat(
            (
                self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]),
                self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1),
            ),
            dim=1,
        )

        token_feat_list = []
        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]

            # NR-IQA 模式下的局部门控池化
            tmp_feat = self.weight_pool[i](tmp_dist_feat)

            if i != end_level - 1:
                # 为 ONNX 导出兼容，使用固定 kernel/stride 的平均池化
                # 替代原始实现中的自适应池化路径。
                downsample = 2 ** ((end_level - 1) - i)
                tmp_feat = F.avg_pool2d(
                    tmp_feat,
                    kernel_size=downsample,
                    stride=downsample,
                    ceil_mode=False,
                    count_include_pad=False,
                )

            tmp_pos_emb = F.interpolate(
                pos_emb, size=tmp_feat.shape[2:], mode='bicubic', align_corners=False
            )
            tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)

            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
            tmp_feat = tmp_feat + tmp_pos_emb

            tmp_feat = self.sa_attn_blks[i](tmp_feat)
            token_feat_list.append(tmp_feat)

        # 从高层到低层，逐步把粗粒度语义传递到细粒度特征
        query = token_feat_list[0]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1]
            query = self.attn_blks[i](query, key_value)

        final_feat = self.attn_pool(query)
        out_score = self.score_linear(final_feat.mean(dim=0))

        return out_score

    def forward(self, x, return_mos=True, return_dist=False):
        """
        前向传播。

        Args:
            x: 输入图像，形状 `(B, 3, H, W)`，值域 `[0, 1]`
            return_mos: 是否返回 MOS 分数
            return_dist: 是否返回 1~10 概率分布

        Returns:
            MOS 分数，或概率分布，或两者组成的列表。
        """
        score = self.forward_cross_attention(x)
        mos = dist_to_mos(score)

        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(score)

        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]


def clean_state_dict(state_dict):
    """清理 checkpoint，移除 DataParallel 产生的 `module.` 前缀。"""
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_topiq_weights(model: CFANet, weight_path: str, device: torch.device) -> None:
    """
    加载 TOPIQ 预训练权重。

    权重格式兼容项目中现有的 pyiqa 导出形式。
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")

    print(_t("logs.topiq_weight_loading", name=os.path.basename(weight_path)))
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)

    if 'params' in state_dict:
        state_dict = state_dict['params']

    state_dict = clean_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(_t("logs.topiq_weight_missing", count=len(missing)))
    if unexpected:
        print(_t("logs.topiq_weight_unexpected", count=len(unexpected)))

    print(_t("logs.topiq_loaded"))


def get_topiq_weight_path():
    """
    获取 TOPIQ 权重文件路径。

    兼容：
    - PyInstaller 打包后的 `_MEIPASS/models`
    - 开发环境下的 `models/`
    """
    weight_name = 'cfanet_iaa_ava_res50-3cd62bb3.pth'

    search_paths = []

    if hasattr(sys, '_MEIPASS'):
        search_paths.append(os.path.join(sys._MEIPASS, 'models', weight_name))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(base_dir, 'models', weight_name))
    search_paths.append(os.path.join(base_dir, weight_name))

    for path in search_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"TOPIQ 权重文件未找到。请确保 models/{weight_name} 存在。\n"
        f"搜索路径: {search_paths}"
    )


class TOPIQScorer:
    """
    TOPIQ 评分封装类。

    提供与旧版评分器兼容的接口，便于在项目中替换实现。
    """

    def __init__(self, device='mps'):
        """
        初始化 TOPIQ 评分器。

        Args:
            device: 计算设备，支持 `mps` / `cuda` / `cpu`
        """
        self.device = self._get_device(device)
        self._model = None

    def _get_device(self, preferred_device='mps'):
        """选择最合适的 torch 设备。"""
        if preferred_device == 'mps':
            try:
                if torch.backends.mps.is_available():
                    return torch.device('mps')
            except:
                pass

        if preferred_device == 'cuda' or torch.cuda.is_available():
            return torch.device('cuda')

        return torch.device('cpu')

    def _load_model(self):
        """延迟加载 TOPIQ 模型和权重。"""
        if self._model is None:
            print(f"正在初始化 TOPIQ 评分器（设备: {self.device}）...")
            weight_path = get_topiq_weight_path()

            self._model = CFANet()
            load_topiq_weights(self._model, weight_path, self.device)
            self._model.to(self.device)
            self._model.eval()

        return self._model

    def calculate_score(self, image_path: str) -> float:
        """
        计算 TOPIQ 美学分数。

        Returns:
            1-10 范围内的美学分数；失败时返回 None。
        """
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return None

        try:
            model = self._load_model()

            # 固定到 384x384，降低显存压力并规避部分设备兼容性问题
            img = Image.open(image_path).convert('RGB')
            img = img.resize((384, 384), Image.LANCZOS)

            transform = T.ToTensor()
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                score = model(img_tensor, return_mos=True)

            if isinstance(score, torch.Tensor):
                score = score.item()

            return float(max(1.0, min(10.0, score)))

        except Exception as e:
            print(f"TOPIQ 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # 简单测试入口
    print("=" * 70)
    print("TOPIQ 独立模型测试")
    print("=" * 70)

    scorer = TOPIQScorer(device='mps')

    test_image = "img/_Z9W0960.jpg"

    if os.path.exists(test_image):
        print(f"\n测试图片: {test_image}")

        import time
        start = time.time()
        score = scorer.calculate_score(test_image)
        elapsed = time.time() - start

        if score is not None:
            print(f"   TOPIQ 分数: {score:.2f} / 10")
            print(f"   耗时: {elapsed*1000:.0f}ms")
        else:
            print("   TOPIQ 计算失败")
    else:
        print(f"\n测试图片不存在: {test_image}")

    print("\n" + "=" * 70)
