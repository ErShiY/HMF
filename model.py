import torch
import torch.nn as nn
import json
import os
import h5py
from utils import BiCrossAttention, ContrastiveLoss, Attention_Gated, T5TextEncoder, Slides_Classifier, PatientPooling, Patient_classifier, text_descriptions_stage
import torch.nn.functional as F


class Model_HMF(nn.Module):
    def __init__(self, h5_path, weight_attention_checkpoint, t5model_path, text_data_path, ratio, subtyping_task=None):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.h5_path = h5_path
        if subtyping_task is None:
            self.cls_weight_camelyon_slides = torch.tensor([0.0529, 0.4678, 0.2853, 0.1940], dtype=torch.float32).cuda()
            self.cls_weight_camelyon_patient = torch.tensor([0.1462, 0.3191, 0.1671, 0.1170, 0.2506], dtype=torch.float32).cuda()
            task = 'cancer-stage'
        elif subtyping_task == 'brca':
            self.cls_weight = torch.tensor([0.2025, 0.7975], dtype=torch.float32).cuda()
            task = subtyping_task
        elif subtyping_task == 'nsclc':
            self.cls_weight = torch.tensor([0.5, 0.5], dtype=torch.float32).cuda()
            task = subtyping_task

        self.models = nn.ModuleDict({
            'weight_attention': Attention_Gated(ratio=ratio, checkpoint_path=weight_attention_checkpoint),
            'text_encoder': T5TextEncoder(model_dir=t5model_path, task=task),
            'BiCrossAttention': BiCrossAttention(embed_dim=512, num_heads=8),
            'ContrastiveLoss': ContrastiveLoss(temperature=0.07),
            'Slides_Classifier': Slides_Classifier(n_cls=4),
            'Patient_Pooling': PatientPooling(input_dim=1024, drop_rate=0.25),
            'Patient_classifier': Patient_classifier(n_cls=5)
        })

        self.proj_1 = nn.LayerNorm(1024)
        self.fusion_proj = nn.Linear(512, 512).to(self.device)
        self.models.to(self.device)
        self.subtyping_task = subtyping_task

        with open(text_data_path, 'r', encoding='utf') as f:
            self.text_data = json.load(f)

    def slide_function(self, features, coords, slide_label, patient_label):

        device = next(self.parameters()).device
        data = self.models['weight_attention'](features, coords)
        use_tensor = data['features']
        use_tensor = self.proj_1(use_tensor)
        del data

        pos_text, neg_text = text_descriptions_stage(patient_label, self.text_data, num_neg=12)
        all_text = pos_text + neg_text
        text_emd = self.models['text_encoder'].encode(all_text)

        attended_text, attended_img, attn_info = self.models['BiCrossAttention'](text_emd, use_tensor)
        # text2img (30,2345)
        # img2text (2345,30)

        del pos_text, neg_text, text_emd, use_tensor

        global_text = self.fusion_proj(attended_text)  # (30,512)
        global_img = self.fusion_proj(attended_img)  # (23456,512)

        global_text = F.normalize(global_text, p=2, dim=1)
        global_img = F.normalize(global_img, p=2, dim=1)
        del attended_text, attended_img, attn_info

        con_loss = self.models['ContrastiveLoss'](global_img, global_text, num_positive=6)

        slide_logits, cla_loss, global_embedding = self.models['Slides_Classifier'](global_img, slide_label, self.cls_weight_camelyon_slides)

        del features, global_img, global_text

        result = {
            'con_loss': con_loss,
            'slide_loss': cla_loss,
            'slide_logits': slide_logits,
            'slide_embedding': global_embedding
        }

        return result

    def patient_function(self, features, patient_label):

        features, attn_weight = self.models['Patient_Pooling'](features)
        patient_logits, loss = self.models['Patient_classifier'](features, patient_label, self.cls_weight_camelyon_patient)

        result = {
            'patient_loss': loss,
            'patient_logits': patient_logits,
        }

        return result

    def slide_only_function(self, features, patient_label):

        device = next(self.parameters()).device
        data = self.models['weight_attention'](features)
        use_tensor = data['features']
        use_tensor = self.proj_1(use_tensor)
        del data

        pos_text, neg_text = text_descriptions_stage(patient_label, self.text_data, self.subtyping_task, num_neg=14)
        all_text = pos_text + neg_text
        text_emd = self.models['text_encoder'].encode(all_text)

        attended_text, attended_img, attn_info = self.models['BiCrossAttention'](text_emd, use_tensor)
        # text2img (30,2345)
        # img2text (2345,30)

        del pos_text, neg_text, text_emd, use_tensor

        global_text = self.fusion_proj(attended_text)  # (30,512)
        global_img = self.fusion_proj(attended_img)  # (23456,512)

        global_text = F.normalize(global_text, p=2, dim=1)
        global_img = F.normalize(global_img, p=2, dim=1)
        del attended_text, attended_img, attn_info

        con_loss = self.models['ContrastiveLoss'](global_img, global_text, num_positive=6)

        global_embedding = self.models['Slides_Classifier'](global_img)
        del features, global_img, global_text

        patient_logits, loss = self.models['Patient_classifier'](global_embedding, patient_label, self.cls_weight)

        result = {
            'con_loss': con_loss,
            'patient_loss': loss,
            'patient_logits': patient_logits
        }

        return result

