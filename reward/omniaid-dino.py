"""Self-contained DINOv3 backbone variant of OmniAID.

Mirrors models/OmniAID_DINO.py with training-only logic (training modes,
gradient checkpointing, orthogonality/balance losses, requires_grad
bookkeeping) removed. Numerical output matches the full implementation
in standard inference mode for the same checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from transformers import DINOv3ViTModel
from transformers.models.dinov3_vit.modular_dinov3_vit import (
    DINOv3ViTEmbeddings,
    DINOv3ViTRopePositionEmbedding,
    DINOv3ViTMLP,
    DINOv3ViTGatedMLP,
    DINOv3ViTLayerScale,
    DINOv3ViTDropPath,
    apply_rotary_pos_emb,
)


class OmniAID_DINO(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.rank_per_expert = config.rank_per_expert
        self.moe_router_hidden_dim = config.moe_router_hidden_dim
        self.top_k = config.moe_top_k

        self.is_hybrid = config.is_hybrid

        if self.is_hybrid:
            self.artifact_expert_idx = config.num_experts - 1
            self.num_semantic_experts = config.num_experts - 1
            gating_num_experts = self.num_semantic_experts
        else:
            self.artifact_expert_idx = -1
            gating_num_experts = self.num_experts

        pretrained_path = config.DINOV3_path

        self.feature_extractor = DINOv3ViTModel.from_pretrained(pretrained_path)
        self.feature_extractor.eval()

        vision_config = self.feature_extractor.config
        self.hidden_size = vision_config.hidden_size

        total_rank = vision_config.hidden_size
        residual_rank = self.rank_per_expert
        r_main = total_rank - residual_rank

        self.embeddings = DINOv3ViTEmbeddings(vision_config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(vision_config)
        self.layer = nn.ModuleList([
            DINOv3ViTMoELayer(
                vision_config, self.num_experts, r_main,
                self.rank_per_expert, self.artifact_expert_idx,
            )
            for _ in range(vision_config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(vision_config.hidden_size, eps=vision_config.layer_norm_eps)
        self.gating_network = GatingNetwork(
            input_dim=self.hidden_size,
            num_experts=gating_num_experts,
            hidden_dim=self.moe_router_hidden_dim,
            top_k=self.top_k,
        )
        self.head = nn.Linear(self.hidden_size, 2)

        self.load_and_replace_from_pretrained(self.feature_extractor)

    def load_and_replace_from_pretrained(self, pretrained_model):
        self.embeddings.load_state_dict(pretrained_model.embeddings.state_dict())
        self.rope_embeddings.load_state_dict(pretrained_model.rope_embeddings.state_dict())
        self.norm.load_state_dict(pretrained_model.norm.state_dict())

        for moe_layer, pretrained_layer in zip(self.layer, pretrained_model.layer):
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                self._replace_linear_with_svd_moe(
                    getattr(pretrained_layer.attention, proj_name),
                    getattr(moe_layer.attention, proj_name),
                )
            moe_layer.mlp.load_state_dict(pretrained_layer.mlp.state_dict())
            moe_layer.norm1.load_state_dict(pretrained_layer.norm1.state_dict())
            moe_layer.norm2.load_state_dict(pretrained_layer.norm2.state_dict())
            moe_layer.layer_scale1.load_state_dict(pretrained_layer.layer_scale1.state_dict())
            moe_layer.layer_scale2.load_state_dict(pretrained_layer.layer_scale2.state_dict())

    def _replace_linear_with_svd_moe(self, original_module: nn.Linear, moe_module):
        original_weight = original_module.weight.data
        U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)
        r = moe_module.r_main
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        moe_module.weight_main.data.copy_(U_r @ torch.diag(S_r) @ Vh_r)

    def forward(self, images) -> dict:
        images = images.to(self.embeddings.patch_embeddings.weight.dtype)
        batch_size = images.size(0)

        with torch.no_grad():
            routing_features = self.feature_extractor(images).pooler_output

        hidden_states = self.embeddings(images)
        position_embeddings = self.rope_embeddings(images)

        if self.is_hybrid:
            artifact_expert_indices = torch.full(
                (batch_size, 1), self.artifact_expert_idx,
                dtype=torch.long, device=hidden_states.device,
            )
            artifact_expert_gates = torch.ones(
                (batch_size, 1), dtype=hidden_states.dtype, device=hidden_states.device,
            )
            semantic_gating_outputs = self.gating_network(routing_features)
            gating_outputs = {
                'top_k_indices': torch.cat(
                    [semantic_gating_outputs['top_k_indices'], artifact_expert_indices], dim=1
                ),
                'top_k_gates': torch.cat(
                    [semantic_gating_outputs['top_k_gates'], artifact_expert_gates], dim=1
                ),
            }
        else:
            gating_outputs = self.gating_network(routing_features)

        final_gates = torch.zeros(
            batch_size, self.num_experts,
            device=hidden_states.device, dtype=gating_outputs['top_k_gates'].dtype,
        )
        final_gates.scatter_(-1, gating_outputs['top_k_indices'], gating_outputs['top_k_gates'])

        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                gating_outputs=gating_outputs,
                position_embeddings=position_embeddings,
            )

        sequence_output = self.norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]
        pred = self.head(pooled_output)
        prob = torch.softmax(pred, dim=1)[:, 1]

        return prob


class DINOv3ViTMoEAttention(nn.Module):
    def __init__(self, config, num_experts, r_main, rank_per_expert, artifact_expert_idx):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx, config.query_bias)
        self.k_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx, config.key_bias)
        self.v_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx, config.value_bias)
        self.o_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx, config.proj_bias)

    def forward(self, hidden_states, gating_outputs, attention_mask=None, position_embeddings=None):
        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, gating_outputs)
        key_states = self.k_proj(hidden_states, gating_outputs)
        value_states = self.v_proj(hidden_states, gating_outputs)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output, gating_outputs)

        return attn_output, attn_weights


class DINOv3ViTMoELayer(nn.Module):
    def __init__(self, config, num_experts, r_main, rank_per_expert, artifact_expert_idx):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTMoEAttention(config, num_experts, r_main, rank_per_expert, artifact_expert_idx)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.drop_path = DINOv3ViTDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_gated_mlp:
            self.mlp = DINOv3ViTGatedMLP(config)
        else:
            self.mlp = DINOv3ViTMLP(config)

        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(self, hidden_states, gating_outputs, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            gating_outputs,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


class SVDMoeLinear(nn.Module):
    def __init__(self, in_features, out_features, r_main, num_experts, rank_per_expert, artifact_expert_idx, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_main = r_main
        self.num_experts = num_experts
        self.rank_per_expert = rank_per_expert
        self.artifact_expert_idx = artifact_expert_idx

        self.register_buffer('weight_main', torch.zeros(out_features, in_features))
        self.register_buffer('U_r', torch.zeros(out_features, r_main))
        self.register_buffer('V_r', torch.zeros(r_main, in_features))

        self.U_experts = nn.ParameterList([nn.Parameter(torch.zeros(out_features, rank_per_expert)) for _ in range(num_experts)])
        self.S_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert)) for _ in range(num_experts)])
        self.V_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert, in_features)) for _ in range(num_experts)])

        self.register_buffer('weight_original_fnorm', torch.tensor(0.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, gating_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_main = F.linear(x, self.weight_main, None)

        top_k_indices = gating_outputs['top_k_indices']
        top_k_gates = gating_outputs['top_k_gates']
        k = top_k_indices.size(1)

        expert_output = torch.zeros_like(output_main)
        U_all = torch.stack([p for p in self.U_experts])
        S_all = torch.stack([p for p in self.S_experts])
        V_all = torch.stack([p for p in self.V_experts])

        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(1)
            expert_output = expert_output.unsqueeze(1)

        for i in range(k):
            chosen_expert_indices = top_k_indices[:, i]
            gate_values = top_k_gates[:, i].unsqueeze(-1)
            U_batch = U_all[chosen_expert_indices]
            S_batch = S_all[chosen_expert_indices]
            V_batch = V_all[chosen_expert_indices]

            x_v = torch.bmm(x, V_batch.transpose(1, 2))
            x_v_s = x_v * S_batch.unsqueeze(1)
            current_expert_output = torch.bmm(x_v_s, U_batch.transpose(1, 2))
            expert_output += current_expert_output * gate_values.unsqueeze(-1)

        if original_dim == 2:
            expert_output = expert_output.squeeze(1)

        final_output = output_main + expert_output
        if self.bias is not None:
            final_output = final_output + self.bias
        return final_output


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.network(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        return {'top_k_indices': top_k_indices, 'top_k_gates': top_k_gates}