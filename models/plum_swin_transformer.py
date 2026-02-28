from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from typing import Optional, Union
import torch.nn as nn
import torch
from transformers import Swinv2Model, AutoConfig
from transformers.models.swinv2.modeling_swinv2 import Swinv2Encoder, Swinv2EncoderOutput, Swinv2ForImageClassification, \
    Swinv2ModelOutput, Swinv2ImageClassifierOutput



class PersistenceDiagramEncoder(nn.Module):
    """
    Persistence Diagram Encoder
    Encodes points (birth, death) in persistence diagram into feature vectors.
    Includes Xavier initialization for linear layers.
    """

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=128):
        super(PersistenceDiagramEncoder, self).__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 6D: birth, death, persistence, birth_death_ratio one-hot
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Apply Xavier initialization to the model's weights
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the model using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, pd_points):
        """
        Args:
            pd_points: [batch_size, num_points, 2] - (birth, death) coordinates
        Returns:
            point_features: [batch_size, num_points, output_dim] - features for each point
            global_features: [batch_size, output_dim] - global features
        """
        batch_size, num_points, _ = pd_points.shape


        birth = pd_points[:, :, 0:1]
        death = pd_points[:, :, 1:2]
        persistence = death - birth

        max_finite_value = 1.0
        persistence = torch.clamp(persistence, min=0.0, max=max_finite_value)
        birth = torch.clamp(birth, min=0.0, max=max_finite_value)
        death = torch.clamp(death, min=0.0, max=max_finite_value)

        epsilon = 1e-2
        birth_safe = torch.clamp(birth, min=epsilon)
        birth_death_ratio = torch.log(death / birth_safe)

        enhanced_features = torch.cat([
            birth, death, persistence, birth_death_ratio, pd_points[:, :, 2:3], pd_points[:, :, 3:4]
            # birth, death, persistence, birth_death_ratio
        ], dim=-1)

        enhanced_features = torch.nan_to_num(enhanced_features, nan=0.0, posinf=max_finite_value, neginf=0.0)

        reshaped_features = enhanced_features.view(-1, enhanced_features.size(-1))

        point_features = self.point_mlp(reshaped_features)
        point_features = point_features.view(batch_size, num_points, -1)

        attended_features, _ = self.attention(point_features, point_features, point_features)

        global_features = torch.mean(attended_features, dim=1)

        return attended_features, global_features


class PlumSwinv2Encoder(Swinv2Encoder):
    def __init__(self, config, modes, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__(config, grid_size, pretrained_window_sizes)
        self.linear_1 = nn.Sequential(nn.Linear(1024, 96), nn.ReLU(), nn.Linear(96, 192))
        self.linear_2 = nn.Sequential(nn.Linear(1024, 196), nn.ReLU(), nn.Linear(196, 384))
        self.linear_3 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 768))
        self.linear_4 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 768))
        self.topology_branch = PersistenceDiagramEncoder(output_dim=1024)
        self.weightTokenEmbedding = ValueTokenEmbedding(192)
        self.sugarTokenEmbedding = ValueTokenEmbedding(192)
        self.norm_weight = nn.LayerNorm(192)
        self.norm_sugar = nn.LayerNorm(192)
        self.modes = modes
        self.w = 0.7
        self.s = 0.3

        self.init_cur_weights()

    def init_cur_weights(self):
        nn.init.xavier_uniform_(self.linear_1[0].weight)
        if self.linear_1[0].bias is not None:
            nn.init.zeros_(self.linear_1[0].bias)

        nn.init.xavier_uniform_(self.linear_2[0].weight)
        if self.linear_2[0].bias is not None:
            nn.init.zeros_(self.linear_2[0].bias)

        nn.init.xavier_uniform_(self.linear_3[0].weight)
        if self.linear_3[0].bias is not None:
            nn.init.zeros_(self.linear_3[0].bias)

        nn.init.xavier_uniform_(self.linear_4[0].weight)
        if self.linear_4[0].bias is not None:
            nn.init.zeros_(self.linear_4[0].bias)

    def forward(
            self,
            hidden_states: torch.Tensor,
            pws: tuple[torch.Tensor],
            input_dimensions: tuple[int, int],
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            output_hidden_states_before_downsampling: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[tuple, Swinv2EncoderOutput]:
        pd, weight, sugar = pws
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape

            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        o = self.topology_branch(pd)[1] if 'persistent' in self.modes else None
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
            )

            x = layer_outputs[0]

            if i == 0:
                weight = self.norm_weight(self.weightTokenEmbedding(weight)).unsqueeze(1).expand_as(x) if 'weight' in self.modes else None
                sugar = self.norm_sugar(self.sugarTokenEmbedding(sugar)).unsqueeze(1).expand_as(x) if 'sugar' in self.modes else None
                # x = x + (self.w * weight if weight is not None else 0) + (self.s * sugar if sugar is not None else 0)
                x = (x + (self.w * weight if weight is not None else 0) + (self.s * sugar if sugar is not None else 0)) / (1 + (self.w if weight is not None else 0) + (self.s if sugar is not None else 0))

            if o is not None:
                o1 = eval(f"self.linear_{i + 1}(o)")
                scale1 = F.sigmoid(o1).unsqueeze(1).expand_as(x)
                out1 = x * scale1 + o1.unsqueeze(1).expand_as(x) + x
                hidden_states = out1
            else:
                hidden_states = x

            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_reshaped_hidden_states]
                if v is not None
            )

        return Swinv2EncoderOutput(
            last_hidden_state=[hidden_states, o],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class PlumSwinModel(Swinv2Model):
    def __init__(self, config, modes, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config, add_pooling_layer, use_mask_token)
        self.encoder = PlumSwinv2Encoder(config=config, modes=modes, grid_size=self.embeddings.patch_grid)

    def forward(
            self,
            x: list[torch.Tensor],
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: bool = False,
            return_dict: Optional[bool] = None,
    ) -> Union[tuple, Swinv2ModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        if isinstance(x, (tuple, list)):
            pixel_values, pd, weight, sugar = x
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            (pd, weight, sugar),
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0][0]
        topo_emb = encoder_outputs[0][1]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        return Swinv2ModelOutput(
            last_hidden_state=[sequence_output, topo_emb],
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )


class PlumSwinv2ForImageClassification(Swinv2ForImageClassification):
    def __init__(self, config, classes_num, modes):
        super().__init__(config)
        self.modes = modes
        self.swinv2 = PlumSwinModel(config, modes)
        self.topo_head = nn.Linear(1024, classes_num)
        self.head = nn.Linear(in_features=768, out_features=classes_num, bias=True)
        self.dropout = nn.Dropout(0.1)
        del self.classifier
        self.init_cur_weights()

    def init_cur_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        nn.init.xavier_uniform_(self.topo_head.weight)
        if self.topo_head.bias is not None:
            nn.init.zeros_(self.topo_head.bias)

    def forward(
            self,
            x: list[torch.Tensor],
            head_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: bool = False,
            return_dict: Optional[bool] = None,
    ) -> Union[tuple, Swinv2ImageClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.swinv2(
            x=x,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        topo_emb = outputs[0][1]
        logits = self.head(pooled_output)

        return logits, self.topo_head(topo_emb) if 'persistent' in self.modes else None


class ValueTokenEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear_w = nn.Linear(in_features=1, out_features=hidden_dim, bias=False)
        self.linear_v = nn.Linear(in_features=1, out_features=hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.linear_w.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        wx = self.linear_w(x)
        vx = self.linear_v(x)

        swish_wx = F.silu(wx)
        h0 = swish_wx * vx

        return h0


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss that considers sample weights
    """

    def __init__(self, alpha=1, gamma=2, class_weights=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets, sample_weights=None):
        # batch_size, seq_len, num_classes = inputs.shape
        # inputs = inputs.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
        # targets = targets.view(-1)  # [batch_size * seq_len]

        # Ensure class_weights is on the same device as inputs
        weight = self.class_weights.to(inputs.device) if self.class_weights is not None else None

        # compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)

        # Focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if sample_weights is not None:
            sample_weights = sample_weights.view(-1)
            focal_loss = focal_loss * sample_weights

        return focal_loss.mean()


def createPeftPlumModel(modes, num_classes=4):
    config = AutoConfig.from_pretrained("./model\swinv2-tiny-patch4-window8-256")
    base_model = PlumSwinv2ForImageClassification(config, classes_num=num_classes, modes=modes)
    pretrained_weights_path = "./model\swinv2-tiny-patch4-window8-256\pytorch_model.bin"
    state_dict = torch.load(pretrained_weights_path, map_location='cpu')
    base_model.load_state_dict(state_dict, strict=False)
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        inference_mode=False,
        modules_to_save=["weightTokenEmbedding", "sugarTokenEmbedding", "topology_branch", "head", "topo_head",
                       "linear_1", "linear_2", "linear_3", "linear_4", "norm_weight", "norm_sugar",],
        # task_type="IMAGE_CLS",
    )
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model




