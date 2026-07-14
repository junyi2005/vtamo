import pdb
import copy
import math
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class AttentionPoolLayer(nn.Module):
    """
    单层注意力池化模块，用于2倍降采样。

    类似于 TemporalConv 中的 K5 + P2，但使用注意力机制替代 MaxPool。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        window_size: int = 8,
        use_relative_pos: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_relative_pos = use_relative_pos
        self.head_dim = hidden_size // num_heads
        self.downsample_rate = 2  # 每层2倍降采样

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # 类似 K5 的卷积
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

        # 注意力投影（替代 MaxPool）
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        # 可学习的查询位置编码
        self.query_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # 相对位置编码
        if use_relative_pos:
            self.max_relative_pos = window_size
            self.relative_pos_embed = nn.Embedding(
                2 * self.max_relative_pos + 1, num_heads
            )

    def forward(
        self,
        x: torch.Tensor,
        lgt: torch.Tensor,
        return_attn: bool = False,
        return_stats: bool = False
    ) -> tuple:
        """
        Args:
            x: [B, D, T] 输入特征
            lgt: [B] 每个样本的有效长度
            return_attn: 是否返回注意力权重

        Returns:
            output: [B, D, T//2] 降采样后的特征
            new_lgt: [B] 新的有效长度
            attn_weights: [B, T//2, T] 注意力权重（仅当 return_attn=True）
        """
        B, D, T = x.shape
        device = x.device

        # Step 1: 卷积（类似 K5）
        x = self.conv(x)  # [B, D, T]

        # 转换为 [B, T, D]
        x = x.permute(0, 2, 1)

        # Step 2: 注意力池化降采样（替代 P2 MaxPool）
        T_out = T // self.downsample_rate
        new_lgt = lgt // self.downsample_rate
        T_out = max(T_out, 1)
        new_lgt = new_lgt.clamp(min=1)

        # 构建注意力矩阵
        if return_attn:
            attn_matrix = torch.zeros(B, T_out, T, device=device, dtype=x.dtype)
        if return_stats:
            peak_sum = torch.tensor(0.0, device=device, dtype=x.dtype)
            entropy_sum = torch.tensor(0.0, device=device, dtype=x.dtype)
            count = torch.tensor(0.0, device=device, dtype=x.dtype)

        outputs = []
        for i in range(T_out):
            # 计算窗口范围
            center = (i + 0.5) * self.downsample_rate
            half_window = self.window_size // 2

            start = max(0, int(center - half_window))
            end = min(T, int(center + half_window))
            window_len = end - start

            if window_len == 0:
                start = min(int(center), T - 1)
                end = start + 1
                window_len = 1

            # 提取窗口
            window = x[:, start:end, :]  # [B, window_len, D]

            # 生成查询
            query_input = window.mean(dim=1, keepdim=True) + self.query_pos_embed

            # 投影
            Q = self.query_proj(query_input)  # [B, 1, D]
            K = self.key_proj(window)  # [B, window_len, D]
            V = self.value_proj(window)  # [B, window_len, D]

            # 多头注意力
            Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, window_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, window_len, self.num_heads, self.head_dim).transpose(1, 2)

            # 注意力分数
            attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

            # 相对位置偏置
            if self.use_relative_pos and window_len <= 2 * self.max_relative_pos + 1:
                center = window_len // 2
                rel_pos_indices = torch.arange(window_len, device=device) - center
                rel_pos_indices = rel_pos_indices.clamp(-self.max_relative_pos, self.max_relative_pos)
                rel_pos_indices = rel_pos_indices + self.max_relative_pos
                rel_bias = self.relative_pos_embed(rel_pos_indices)
                rel_bias = rel_bias.transpose(0, 1).unsqueeze(0).unsqueeze(2)
                attn_scores = attn_scores + rel_bias

            # 有效长度掩码
            frame_indices = torch.arange(start, end, device=device).unsqueeze(0)
            valid_mask = frame_indices < lgt.unsqueeze(1)
            valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~valid_mask, float('-inf'))

            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / window_len)

            # 保存注意力权重
            if return_attn:
                avg_attn = attn_weights.mean(dim=1).squeeze(1)
                attn_matrix[:, i, start:end] = avg_attn
            if return_stats:
                with torch.no_grad():
                    attn = attn_weights.squeeze(2)  # [B, heads, window_len]
                    peak = attn.max(dim=-1).values
                    if window_len > 1:
                        ent = -(attn * attn.clamp(min=1e-8).log()).sum(dim=-1) / math.log(window_len)
                    else:
                        ent = torch.zeros_like(peak)

                    peak_sum += peak.sum()
                    entropy_sum += ent.sum()
                    count += peak.numel()

            # 加权求和
            out = torch.matmul(attn_weights, V)
            out = out.transpose(1, 2).contiguous().view(B, 1, D)
            outputs.append(out)

        # 拼接并投影
        output = torch.cat(outputs, dim=1)  # [B, T_out, D]
        output = self.output_proj(output)

        # 转换回 [B, D, T_out]
        output = output.permute(0, 2, 1)

        stats = None
        if return_stats:
            stats = {
                "peak_sum": peak_sum,
                "entropy_sum": entropy_sum,
                "count": count,
            }

        if return_attn:
            return output, new_lgt, attn_matrix, stats
        return output, new_lgt, None, stats


class AttentionTemporalConv(nn.Module):
    """
    可学习的注意力时序降采样模块 - 替代 TemporalConv

    两层结构，类似于 TemporalConv 的 K5+P2+K5+P2：
    - Layer 1: Conv + AttentionPool (2倍降采样)
    - Layer 2: Conv + AttentionPool (2倍降采样)
    - 总共 4 倍降采样

    核心区别：
    - TemporalConv: 固定 kernel size 的卷积 + MaxPool 降采样
      - 感受野固定（K5 + P2 + K5 + P2 → 每个输出帧固定看约16帧）
      - MaxPool 不可学习，只选最大值
      - 梯度只能传到被MaxPool选中的位置

    - AttentionTemporalConv: 可学习的注意力降采样
      - 每个输出位置用注意力关注输入帧
      - 感受野由注意力权重动态决定，可学习
      - 梯度可以反传到所有输入帧（通过 softmax 权重）

    输入: [B, D, T_orig] 原始帧特征 (与 TemporalConv 一致)
    输出: [B, D, T_out] 降采样后特征，T_out ≈ T_orig / 4
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        downsample_rate: int = 4,  # 总降采样率（必须是2的幂次）
        num_heads: int = 4,
        window_size: int = 8,  # 每层的窗口大小
        use_relative_pos: bool = True,
        dropout: float = 0.1,
        num_classes: int = -1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.downsample_rate = downsample_rate
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_relative_pos = use_relative_pos
        self.num_classes = num_classes

        # 计算需要多少层（每层2倍降采样）
        self.num_layers = int(math.log2(downsample_rate))
        assert 2 ** self.num_layers == downsample_rate, "downsample_rate must be power of 2"

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

        # 多层注意力池化
        self.layers = nn.ModuleList([
            AttentionPoolLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                window_size=window_size,
                use_relative_pos=use_relative_pos,
                dropout=dropout,
            )
            for _ in range(self.num_layers)
        ])

        # 可选的分类头
        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(
        self,
        frame_feat: torch.Tensor,
        lgt: torch.Tensor,
        return_attn: bool = False,
        return_attn_stats: bool = False
    ):
        """
        Args:
            frame_feat: [B, D, T] 原始帧特征（与 TemporalConv 接口一致）
            lgt: [B] 每个样本的有效帧长度
            return_attn: 是否返回注意力权重（用于反推原始帧对应关系）

        Returns:
            dict with:
                visual_feat: [T_out, B, D] 降采样后的帧特征
                conv_logits: 分类 logits（如果 num_classes > 0）
                feat_len: 降采样后的有效长度
                attn_weights: [B, T_out, T_orig] 注意力权重矩阵（仅当 return_attn=True）
                              attn_weights[b, m, t] = 降采样帧 m 对原始帧 t 的注意力权重
                layer_attn_weights: List of [B, T_i, T_{i-1}] 每层的注意力权重
        """
        B, D, T_orig = frame_feat.shape
        device = frame_feat.device

        # Step 1: 输入投影 [B, D, T] -> [B, hidden_size, T]
        x = self.input_proj(frame_feat)

        # Step 2: 逐层注意力池化降采样
        current_lgt = lgt.clone()
        layer_attn_list = []

        peak_sum = None
        entropy_sum = None
        count_sum = None
        layer_peak_means = []
        layer_entropy_means = []
        for layer_idx, layer in enumerate(self.layers):
            x, current_lgt, layer_attn, layer_stats = layer(
                x,
                current_lgt,
                return_attn=return_attn,
                return_stats=return_attn_stats
            )
            if return_attn:
                layer_attn_list.append(layer_attn)
            if return_attn_stats and layer_stats is not None:
                layer_count = layer_stats["count"]
                if layer_count.item() > 0:
                    layer_peak_mean = (layer_stats["peak_sum"] / layer_count).detach()
                    layer_entropy_mean = (layer_stats["entropy_sum"] / layer_count).detach()
                    layer_peak_means.append(layer_peak_mean)
                    layer_entropy_means.append(layer_entropy_mean)
                if peak_sum is None:
                    peak_sum = layer_stats["peak_sum"]
                    entropy_sum = layer_stats["entropy_sum"]
                    count_sum = layer_stats["count"]
                else:
                    peak_sum = peak_sum + layer_stats["peak_sum"]
                    entropy_sum = entropy_sum + layer_stats["entropy_sum"]
                    count_sum = count_sum + layer_stats["count"]

        # x: [B, D, T_out]
        # 转换为 [T_out, B, D]（与 TemporalConv 输出格式一致）
        visual_feat = x.permute(2, 0, 1)

        # 可选的分类头
        logits = None
        if self.num_classes != -1:
            logits = self.fc(x.permute(0, 2, 1))  # [B, T_out, num_classes]
            logits = logits.permute(1, 0, 2)  # [T_out, B, num_classes]

        result = {
            "visual_feat": visual_feat,
            "conv_logits": logits,
            "feat_len": current_lgt.cpu(),
        }

        # 添加注意力信息（用于反推原始帧对应关系）
        if return_attn and len(layer_attn_list) > 0:
            # 链式乘法计算最终的注意力矩阵
            # A_total = A_layer2 @ A_layer1
            # 这样 A_total[b, m, t] 表示最终降采样帧 m 对原始帧 t 的总注意力
            combined_attn = layer_attn_list[0]  # [B, T1, T_orig]
            for i in range(1, len(layer_attn_list)):
                # layer_attn_list[i]: [B, T_{i+1}, T_i]
                # combined_attn: [B, T_i, T_orig]
                # 结果: [B, T_{i+1}, T_orig]
                combined_attn = torch.bmm(layer_attn_list[i], combined_attn)

            result["attn_weights"] = combined_attn  # [B, T_out, T_orig]
            result["layer_attn_weights"] = layer_attn_list  # 每层的注意力
        if return_attn_stats and count_sum is not None and count_sum.item() > 0:
            result["attn_peak_mean"] = (peak_sum / count_sum).detach()
            result["attn_entropy_mean"] = (entropy_sum / count_sum).detach()
            result["layer_attn_peak_mean"] = layer_peak_means
            result["layer_attn_entropy_mean"] = layer_entropy_means

        return result

    def update_lgt(self, lgt: torch.Tensor) -> torch.Tensor:
        """计算降采样后的长度（与 TemporalConv 接口兼容）"""
        return (lgt // self.downsample_rate).clamp(min=1)


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                    #MultiScale_TemporalConv(input_sz, self.hidden_size)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                #pass
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,
            "feat_len": lgt.cpu(),
        }
    

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, stride=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + residual  # Element-wise addition
        out = self.relu(out)
        return out


class GlorTemporalConv(nn.Module):
    def __init__(self, input_channels, output_channels, dilation_rate=1):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)
        )
        self.layers.append(ResidualBlock(output_channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 2, 1)


def compute_frame_to_token_alignment(
    attn_weights: torch.Tensor,
    alignment_matrix: torch.Tensor,
    text_mask: torch.Tensor = None,
) -> dict:
    """
    反推原始帧和 text token 的对应关系。

    通过链式关系计算：
        P(原始帧 t 对应 token k) = Σ_m attn_weights[m, t] × alignment[m, k]

    Args:
        attn_weights: [B, T_out, T_orig] 降采样注意力权重
                      attn_weights[b, m, t] = 降采样帧 m 对原始帧 t 的注意力
        alignment_matrix: [B, T_out, K] OT 对齐矩阵
                          alignment[b, m, k] = 降采样帧 m 对应 token k 的概率
        text_mask: [B, K] 可选的文本掩码

    Returns:
        dict with:
            frame_token_prob: [B, T_orig, K] 原始帧对应各 token 的概率
                              frame_token_prob[b, t, k] = P(帧 t 对应 token k)
            frame_token_assignment: [B, T_orig] 每个原始帧对应的 token (argmax)
            token_frame_ranges: List[List[Tuple[int, int]]] 每个 token 对应的帧范围
                                token_frame_ranges[b][k] = (start_frame, end_frame)
    """
    B, T_out, T_orig = attn_weights.shape
    _, _, K = alignment_matrix.shape

    # 计算原始帧到 token 的概率
    # frame_token_prob[b, t, k] = Σ_m attn_weights[b, m, t] × alignment[b, m, k]
    # attn_weights: [B, T_out, T_orig] -> [B, T_orig, T_out]
    # alignment: [B, T_out, K]
    # 结果: [B, T_orig, K]
    attn_weights_transposed = attn_weights.transpose(1, 2)  # [B, T_orig, T_out]
    frame_token_prob = torch.bmm(attn_weights_transposed, alignment_matrix)  # [B, T_orig, K]

    # 归一化（可选，让每帧的概率和为1）
    frame_token_prob = frame_token_prob / (frame_token_prob.sum(dim=-1, keepdim=True) + 1e-8)

    # 每帧对应的 token (argmax)
    frame_token_assignment = frame_token_prob.argmax(dim=-1)  # [B, T_orig]

    # 计算每个 token 对应的帧范围
    token_frame_ranges = []
    for b in range(B):
        batch_ranges = []
        for k in range(K):
            # 找到该 token 的所有帧
            token_mask = (frame_token_assignment[b] == k)
            if token_mask.any():
                indices = torch.where(token_mask)[0]
                start_frame = indices.min().item()
                end_frame = indices.max().item() + 1  # exclusive end
            else:
                start_frame = -1
                end_frame = -1
            batch_ranges.append((start_frame, end_frame))
        token_frame_ranges.append(batch_ranges)

    return {
        "frame_token_prob": frame_token_prob,  # [B, T_orig, K]
        "frame_token_assignment": frame_token_assignment,  # [B, T_orig]
        "token_frame_ranges": token_frame_ranges,  # List[List[Tuple[int, int]]]
    }


def visualize_frame_token_alignment(
    frame_token_prob: torch.Tensor,
    token_texts: list = None,
    sample_idx: int = 0,
    save_path: str = None,
):
    """
    可视化原始帧和 token 的对应关系。

    Args:
        frame_token_prob: [B, T_orig, K] 原始帧对应各 token 的概率
        token_texts: 可选的 token 文本列表
        sample_idx: 要可视化的样本索引
        save_path: 保存路径（如果为 None 则显示）
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, cannot visualize")
        return

    prob = frame_token_prob[sample_idx].detach().cpu().numpy()  # [T_orig, K]
    T_orig, K = prob.shape

    fig, ax = plt.subplots(figsize=(max(12, K * 0.5), max(8, T_orig * 0.05)))

    # 绘制热力图
    im = ax.imshow(prob, aspect='auto', cmap='viridis')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Original Frame Index')
    ax.set_title('Frame-to-Token Alignment Probability')

    # 添加 colorbar
    plt.colorbar(im, ax=ax, label='Probability')

    # 如果有 token 文本，添加标签
    if token_texts is not None and len(token_texts) >= K:
        ax.set_xticks(range(K))
        ax.set_xticklabels(token_texts[:K], rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()
