import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import MultiheadAttention
from transformers import T5Tokenizer, T5EncoderModel
import random

anti_patient_label_mapping = {
    0: "pN0",
    1: "pN0(i+)",
    2: "pN1mi",
    3: "pN1",
    4: "pN2"
}

anti_slide_subtype_mapping_nsclc = {
    0: "LUAD",
    1: "LUSC"
}

anti_slide_subtype_mapping_brca = {
    0: "IDC",
    1: "ILC"
}


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.layers_a = [nn.Linear(L, D), nn.Tanh()]
        self.layers_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.layers_a.append(nn.Dropout(0.25))
            self.layers_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.layers_a)
        self.attention_b = nn.Sequential(*self.layers_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class Get_scores(nn.Module):
    def __init__(self, dropout=0., embed_dim=1024):
        super().__init__()
        size = [embed_dim, 512, 256]
        fc = [
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=(dropout > 0), n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

    def forward(self, h):
        A, h = self.attention_net(h)
        return A


class Attention_Gated(nn.Module):
    def __init__(self, ratio=0.8, checkpoint_path='./checkpoints'):
        super().__init__()
        self.ratio = ratio
        self.checkpoint_path = checkpoint_path

        # 只在初始化时加载模型
        if not hasattr(self, 'model_getscores'):  # 检查模型是否已经加载
            self.model_getscores = Get_scores(dropout=0.25, embed_dim=1024)
            ckpt = torch.load(self.checkpoint_path)
            ckpt_clean = {}
            for key in ckpt.keys():
                if 'instance_loss_fn' in key:
                    continue
                ckpt_clean.update({key.replace('.module', ''): ckpt[key]})

            self.model_getscores.load_state_dict(ckpt_clean, strict=False)
            self.model_getscores.eval()

    def forward(self, features, coords=None):
        with torch.no_grad():
            scores = self.model_getscores(features)
            length = int(len(scores))
            scores = torch.transpose(scores, 1, 0)
            scores = F.softmax(scores, dim=1)

            # inst_eval, k mean useful n mean useless
            k = int(length * self.ratio)

            top_p_ids = torch.topk(scores, k=k)[1][-1]

            top_p_features = torch.index_select(features, dim=0, index=top_p_ids)
            if coords is not None:
                top_p_coords = torch.index_select(coords, dim=0, index=top_p_ids)
                return {"coords": top_p_coords, "features": top_p_features}
            else:
                return {"features": top_p_features}


class T5TextEncoder(nn.Module):
    def __init__(self, model_dir: str = "./my_local_t5_model/", task='cancer-stage'):
        super().__init__()

        def tokens_already_added(tokenizer, new_tokens):
            return all(tok in tokenizer.get_vocab() for tok in new_tokens)

        # 检查模型是否已加载
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            # 只在初始化时加载模型
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
            self.model = T5EncoderModel.from_pretrained(model_dir)
            task_model_dir = f"./t5_model_{task.replace('-', '_')}/"

            # 扩展 tokenizer 的词汇
            # cancer-stage
            if task == 'cancer-stage':
                new_tokens = [
                    "macro-metastatic", "Histopathological", "micrometastatic", "Micrometastases",
                    "node-positive", "cytokeratin-positive", "isolated-cell", "microenvironment",
                    "dissemination", "classification", "architectural", "Comprehensive", "pN0"
                ]
            elif task == 'brca':
                new_tokens = [
                    "duct-like", "collagen-rich", "nuclear-cytoplasmic", "single-file",
                    "E-cadherin", "inconspicuous", "low-grade"
                ]
            elif task == 'nsclc':
                new_tokens = [
                    "macro-metastatic", "Histopathological", "micrometastatic", "Micrometastases",
                    "node-positive", "cytokeratin-positive", "isolated-cell", "microenvironment",
                    "dissemination", "classification", "architectural", "Comprehensive", "pN0",
                    "adenocarcinoma", "Lymphovascular", "gland-forming", "Early-stage", "intercellular",
                    "Immunohistochemistry", "p40", "p63", "CK5/6", "responsiveness", "CT-guided", "comorbidities"
                ]

            if not tokens_already_added(self.tokenizer, new_tokens):
                added = self.tokenizer.add_tokens(new_tokens)
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.tokenizer.save_pretrained(task_model_dir, legacy=False)

    def encode(self, text: list[str]):
        # Tokenize and prepare input IDs
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = outputs.last_hidden_state  # [B, T, 512]

        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        masked_hidden = last_hidden * mask  # [B, T, 512]
        sum_hidden = masked_hidden.sum(dim=1)  # [B, 512]
        lengths = mask.sum(dim=1)  # [B, 1]
        mean_pooled = sum_hidden / lengths  # [B, 512]

        return mean_pooled  # shape: (5, 512)


def text_descriptions_stage(label, text_data, task='camelyon', num_neg=12):
    label_value = label.item()

    if task == 'camelyon':
        label_mapped = anti_patient_label_mapping.get(label_value)
    elif task == 'nsclc':
        label_mapped = anti_slide_subtype_mapping_nsclc.get(label_value)
    elif task == 'brca':
        label_mapped = anti_slide_subtype_mapping_brca.get(label_value)

    if label_mapped is None:
        raise ValueError(f"Label value {label_value} not found in mapping for task {task}")

    all_labels = list(text_data.keys())

    pos_pool = text_data[label_mapped]

    neg_pool_all = []
    for other_label in all_labels:
        if other_label != label_mapped:
            neg_pool_all.extend([
                f"{text}"
                for text in text_data[other_label]
            ])

    neg_pool = random.sample(neg_pool_all, min(num_neg, len(neg_pool_all)))

    return pos_pool, neg_pool


class BiCrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.25):
        super().__init__()
        self.pro_layer_img = nn.Linear(1024, embed_dim)
        self.pro_layer_text = nn.Linear(embed_dim, embed_dim)

        self.mha_text2img = MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.mha_img2text = MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

        # LayerNorm 或其他后处理
        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_img = nn.LayerNorm(embed_dim)

    def forward(self, text_dataset, img_dataset):
        """
        text_embeds: (B, M, d)
        img_embeds:  (B, K, d)
        返回:
          attended_text: (B, M, d)
          attended_img:  (B, K, d)
          attn_weights_text2img: (B, M, K)
          attn_weights_img2text: (B, K, M)
        """
        img_embeds = self.pro_layer_img(img_dataset)  # (2345,512)
        text_embeds = self.pro_layer_text(text_dataset)  # (6,512)

        # 1) 文本->图像 注意力
        #    Query=text_embeds, Key=img_embeds, Value=img_embeds
        #    out_text: (B, M, d)
        out_text, attn_weights_text2img = self.mha_text2img(
            query=text_embeds,
            key=img_embeds,
            value=img_embeds
        )
        # 残差 + LayerNorm
        out_text = self.norm_text(out_text + text_embeds)

        # 2) 图像->文本 注意力
        #    Query=img_embeds, Key=out_text, Value=out_text
        #    out_img: (B, K, d)
        out_img, attn_weights_img2text = self.mha_img2text(
            query=img_embeds,
            key=out_text,
            value=out_text
        )
        out_img = self.norm_img(out_img + img_embeds)

        # 5) 打包注意力信息
        attn_info = {
            "text2img": attn_weights_text2img,  # (B, M, K)
            "img2text": attn_weights_img2text,  # (B, K, M)
        }

        return out_text, out_img, attn_info


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.img_proj = nn.Linear(512, 512)
        self.txt_proj = nn.Linear(512, 512)

    def forward(self, global_img, global_text, num_positive=6):
        """
        global_img:   [N_patch, 512] [2345,512]
        global_text:  [N_pseudo, 512] [6,512]
        positive_mask: FloatTensor [B] → 1 for positive, 0 for negative
        """
        img_feat = global_img.mean(dim=0, keepdim=True)  # [1, 512]
        total = global_text.size(0)

        positive_mask = torch.zeros(total, device=img_feat.device)
        positive_mask[:num_positive] = 1.0

        img_feat = F.normalize(self.img_proj(img_feat), dim=-1)  # [1, 512]
        text_feat = F.normalize(self.txt_proj(global_text), dim=-1)  # [6, 512]

        # image contrastive loss
        sim = torch.matmul(img_feat, text_feat.T) / self.temperature  # [1, 6]
        soft_label = positive_mask / (positive_mask.sum() + 1e-6)  # [6]
        soft_label = soft_label.unsqueeze(0)  # [1, 6]
        log_probs = F.log_softmax(sim, dim=1)  # [1, 6]
        loss_img = F.kl_div(log_probs, soft_label, reduction="batchmean")

        # text contrastive loss
        img_feat2text = img_feat.squeeze(0)  # [512]
        sim_text = torch.matmul(text_feat, img_feat2text.unsqueeze(-1)).squeeze(-1) / self.temperature  # [6]
        labels_text = torch.zeros(total, device=sim_text.device)
        labels_text[:num_positive] = 1.0
        loss_text = F.binary_cross_entropy_with_logits(sim_text, labels_text)

        loss = (loss_img + loss_text) / 2
        return loss


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=1024, n_head=8, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        # 定义投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size, _ = query.size()

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 上下文聚合
        context = torch.matmul(attn, V)  # [batch, n_head, d_k]

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1)

        # 最终投影
        output = self.W_o(context)
        importance_scores = attn.sum(dim=1).mean(dim=-1)  # 可以通过累加每个头的注意力得分来获取重要性得分

        return output, importance_scores


class Slides_Classifier(nn.Module):
    def __init__(self, drop_rate=0.3, n_cls=4, smoothing=0.1, gamma=2):
        super(Slides_Classifier, self).__init__()

        self.MHA = MultiHeadAttention(d_model=512, n_head=8, dropout=drop_rate)
        self.ce_para = nn.Parameter(torch.zeros(1))
        self.focal_para = nn.Parameter(torch.zeros(1))

        self.ffn = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(drop_rate),
            nn.Linear(128, 512)
        )

        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024, batch_first=True),
            num_layers=2
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.LayerNorm(1024)
        )
        self.num_classes = n_cls
        self.classifier = nn.Linear(1024, n_cls)
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.gamma = gamma

    def forward(self, x, label=None, cls_weight=None):
        attn_out, importance_scores = self.MHA(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        x = self.encoder(x)
        x = x.transpose(1, 0)
        x = self.pool(x).squeeze(-1)
        pooled = self.out_proj(x).squeeze(0)

        if label is not None:
            logits = self.classifier(pooled)
            ce_loss = self.ce_loss_fn(logits, label)

            log_pt = F.log_softmax(logits, dim=-1)
            pt = torch.exp(log_pt)

            weight_t = cls_weight[label]
            pt_t = pt[label]
            focal_loss = -weight_t * (1 - pt_t) ** self.gamma * log_pt[label]

            loss = (
                    torch.exp(-self.ce_para) * ce_loss + self.ce_para +
                    torch.exp(-self.focal_para) * focal_loss + self.focal_para
            )
            return logits, loss.squeeze(), pooled

        else:
            return pooled


class PatientPooling(nn.Module):
    def __init__(self, input_dim=1024, drop_rate=0.3):
        super(PatientPooling, self).__init__()

        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        attn_scores = self.attention_pool(x)
        attn_weights = F.softmax(attn_scores, dim=0)

        pooled = torch.sum(attn_weights * x, dim=0, keepdim=True)

        return pooled, attn_weights


class Patient_classifier(nn.Module):
    def __init__(self, drop_rate=0.3, n_cls=5, input_dim=1024, num_heads=8, smoothing=0.1, gamma=2):
        super(Patient_classifier, self).__init__()

        self.input_dim = input_dim
        self.n_cls = n_cls
        self.smoothing = smoothing

        self.positional_encoding = nn.Parameter(torch.randn(5, input_dim))
        self.ce_para = nn.Parameter(torch.zeros(1))
        self.focal_para = nn.Parameter(torch.zeros(1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=drop_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc_res1 = nn.Linear(input_dim, 256)
        self.fc_res2 = nn.Linear(256, input_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, n_cls)
        )

        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.gamma = gamma

    def forward(self, x, label=None, cls_weight=None):

        x = x + self.positional_encoding.unsqueeze(0)
        identity = x

        x = self.transformer_encoder(x)
        x = x + identity

        x_pooled = x.mean(dim=1)

        res = x_pooled
        x_res = self.fc_res1(x_pooled)
        x_res = F.relu(x_res)
        x_res = self.fc_res2(x_res)

        x_res = (x_res + res).squeeze(0)

        logits = self.classifier(x_res)
        ce_loss = self.ce_loss_fn(logits, label)

        log_pt = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_pt)

        weight_t = cls_weight[label]
        pt_t = pt[label]
        focal_loss = -weight_t * (1 - pt_t) ** self.gamma * log_pt[label]

        loss = (
                torch.exp(-self.ce_para) * ce_loss + self.ce_para +
                torch.exp(-self.focal_para) * focal_loss + self.focal_para
        )

        return logits, loss


def setup_logger(log_file="logs/train.log", logger_name="train_logger"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 文件输出
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台输出
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"🔁 EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"💾 New best model saved with score: {self.best_score:.4f} at {self.save_path}")

