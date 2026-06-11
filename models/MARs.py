import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 0. Positional Encoding (변경 없음)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

# ==============================================================================
# ### NEW ### 1. Bottleneck 구조를 가진 입력 프로젝션 모듈
# ==============================================================================
class BottleneckProjection(nn.Module):
    """ 병목(Bottleneck) 구조를 가진 MLP 프로젝션 """
    def __init__(self, input_dim, model_dim, bottleneck_dim=512, dropout=0.1):
        super().__init__()
        # 1. 입력 차원에서 병목 차원으로 압축
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 2. 병목 차원에서 최종 모델 차원으로 확장
        self.fc2 = nn.Linear(bottleneck_dim, model_dim)

    def forward(self, x):
        # x: (B, T, D_input)
        x = self.fc1(x)         # -> (B, T, bottleneck_dim)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)         # -> (B, T, D_model)
        return x

# ==============================================================================
# New module: FrameDifferenceEmbedding
# ==============================================================================
class FrameDifferenceEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, T, D)
        diff = x[:, 1:, :] - x[:, :-1, :]
        diff = F.pad(diff, (0, 0, 1, 0))  # pad at beginning to keep shape
        return self.linear(diff)

# ==============================================================================
# 2. MARs의 핵심 구성 모듈들 (변경 없음)
# ==============================================================================

class FeedForwardModule(nn.Module):
    """ 피드 포워드 모듈 (Macaron-Net 스타일의 Half-step Residual) """
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.swish = nn.SiLU() # Swish 활성화 함수
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x_norm = self.layer_norm(x)
        x_ff = self.linear2(self.dropout1(self.swish(self.linear1(x_norm))))
        # 논문에서 제안한 half-step residual connection (0.5를 곱함)
        return x + 0.5 * self.dropout2(x_ff)

class MultiHeadSelfAttentionModule(nn.Module):
    """ 멀티 헤드 셀프 어텐션 모듈 """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x_norm = self.layer_norm(x)
        # self-attention
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        return x + self.dropout(attn_output)

class ConvolutionModule(nn.Module):
    """
    Depthwise Convolution Module with parallel multi‑kernel branches.
    Pass a tuple/list of temporal kernel sizes via ``kernel_sizes``.
    """
    def __init__(self, d_model, kernel_sizes=(31,), dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        # 1×1 point‑wise expansion + GLU gate
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)

        # Depth‑wise convolutions (time axis) for each kernel size
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(d_model,
                      d_model,
                      ks,
                      padding=(ks - 1) // 2,
                      groups=d_model)
            for ks in kernel_sizes
        ])

        # Batch‑norm + activation after concatenation
        self.batch_norm = nn.BatchNorm1d(d_model * len(kernel_sizes))
        self.swish = nn.SiLU()

        # Fuse back to d_model channels
        self.pointwise_conv2 = nn.Conv1d(d_model * len(kernel_sizes),
                                         d_model,
                                         1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : (B, T, D)
        x_ln = self.layer_norm(x).transpose(1, 2)        # → (B, D, T)
        x_pw = self.glu(self.pointwise_conv1(x_ln))      # (B, D, T)

        # Parallel branch processing
        branch_outs = [conv(x_pw) for conv in self.dw_convs]  # list[(B,D,T)]
        x_cat = torch.cat(branch_outs, dim=1)            # (B, D*branches, T)

        x_cat = self.swish(self.batch_norm(x_cat))
        x_cat = self.pointwise_conv2(x_cat)              # (B, D, T)
        x_cat = self.dropout(x_cat)

        return x + x_cat.transpose(1, 2)                 # back to (B, T, D)

# ==============================================================================
# 3. MARs 블록 (변경 없음)
# ==============================================================================
class MARsBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_expansion_factor=4, conv_kernel_sizes=(31,), dropout=0.1, order="att_first"):
        super().__init__()
        assert order in ["att_first", "conv_first"], "order must be 'att_first' or 'conv_first'"
        self.order = order

        self.ff_module1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.attention_module = MultiHeadSelfAttentionModule(d_model, nhead, dropout)
        self.conv_module = ConvolutionModule(d_model, kernel_sizes=conv_kernel_sizes, dropout=dropout)
        self.ff_module2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ff_module1(x)

        if self.order == "att_first":
            x = self.attention_module(x)
            x = self.conv_module(x)
        else: # "conv_first"
            x = self.conv_module(x)
            x = self.attention_module(x)

        x = self.ff_module2(x)
        x = self.final_layer_norm(x)
        return x

# ==============================================================================
# 4. 최종 비디오 요약 모델 (Bottleneck Projection 적용)
# ==============================================================================
class MARs(nn.Module):
    def __init__(self, input_dim, model_dim=256, nhead=4, num_blocks=4,
                 conv_kernel_sizes=(31,), dropout=0.1, order="att_first"):
        super().__init__()

        # ### MODIFIED ###: Bottleneck 프로젝션 모듈 사용
        self.input_projection = BottleneckProjection(
            input_dim=input_dim,
            model_dim=model_dim,
            bottleneck_dim= 128, #use the original 256 instead of the 128 experiment value, # 병목 차원 크기는 조절 가능한 하이퍼파라미터
            dropout=dropout
        )
        self.frame_diff = FrameDifferenceEmbedding(model_dim)
        
        self.pos_encoder = PositionalEncoding(model_dim)

        self.blocks = nn.ModuleList([
            MARsBlock(
                d_model=model_dim,
                nhead=nhead,
                conv_kernel_sizes=conv_kernel_sizes,
                dropout=dropout,
                order=order
            ) for _ in range(num_blocks)
        ])
        
        self.linear1 = nn.Linear(model_dim, model_dim // 2)
        self.linear2 = nn.Linear(model_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout_output = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # - The feature vectors are now received as (1, T, D), NO Need to discard the 2-channels

        # if x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == 3:
        #     x = x.squeeze(0)[0] 
        #     x = x.unsqueeze(0)

        # x shape: (batch_size, seq_len, input_dim)
        
        # 1. 입력 프로젝션: Bottleneck MLP를 사용하여 피쳐 차원 변환
        x = self.input_projection(x)
        x = x + self.frame_diff(x)  # Stop using FrameDifferenceEmbedding 
        # x shape: (batch_size, seq_len, model_dim)

        # 2. Positional Encoding 적용
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # 3. MARs 블록 통과
        for block in self.blocks:
            x = block(x)
        
        # 4. 깊은 최종 출력 레이어
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout_output(x)
        x = self.linear2(x)
        
        # 5. 시그모이드 활성화: 점수를 0~1 사이로 변환
        scores = self.sigmoid(x)
        
        return scores.squeeze()


if __name__ == '__main__':
    # ==========================================================================
    # 모델 사용 예시
    # ==========================================================================
    
    T = 1200
    D = 1024
    dummy_feature = torch.randn(T, D)
    dummy_input_tensor = dummy_feature.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
    print(f"입력 텐서 모양: {dummy_input_tensor.shape}")

    # --- Bottleneck Input Projection이 적용된 모델 생성 ---
    print("\n--- Bottleneck 적용 모델 ---")
    model_with_bottleneck = MARs(
        input_dim=D,
        model_dim=256,
        nhead=4,
        num_blocks=4,
        conv_kernel_sizes=(15, 31, 63),
        order="att_first"
    )
    
    # 모델 파라미터 수 확인 (참고용)
    total_params = sum(p.numel() for p in model_with_bottleneck.parameters() if p.requires_grad)
    print(f"모델의 총 파라미터 수: {total_params:,}")

    # 모델에 입력하여 결과 확인
    output_scores = model_with_bottleneck(dummy_input_tensor)
    print(f"출력 점수 모양: {output_scores.shape}")
    print(f"출력 점수 예시 (처음 5개): {output_scores[:5]}")

