import glob

import h5py
from torch.utils.data import Dataset
import os
import pandas as pd
import torch


class CSVReader:
    """Class to read and process CSV files for training, validation, and testing splits."""

    def __init__(self, csv_dir, fold):
        self.csv_dir = csv_dir
        self.fold = fold
        self.csv_files = {
            'train': os.path.join(csv_dir, f'train_{fold}.csv'),
            'test': os.path.join(csv_dir, f'test_{fold}.csv'),
            'validation': os.path.join(csv_dir, f'val_{fold}.csv')
        }

    def __getitem__(self, split):
        if split not in self.csv_files:
            raise ValueError(
                f"Invalid split name: {split}. Must be one of {list(self.csv_files.keys())}."
            )

        # Read the CSV
        df = pd.read_csv(self.csv_files[split], dtype={'patient_id_num': str})

        # Extract relevant columns
        slides = df['patient_id_num'].tolist()
        slide_labels = df['slide_stage'].tolist()
        patient_labels = df['patient_stage'].tolist()

        dataset = {
            'slides': slides,
            'slide_labels': slide_labels,
            'patient_labels': patient_labels
        }
        datasets = [{"slide": slide, "slide_label": slide_label, "patient_label": patient_label} for
                    slide, slide_label, patient_label in
                    zip(dataset["slides"], dataset["slide_labels"], dataset["patient_labels"])]
        # datasets = pd.DataFrame(datasets)
        return datasets


class CSVReaderTCGA:
    """读取 TCGA 的 train_1.csv / val_1.csv 文件，提取 patient_id 和 subtype"""

    def __init__(self, csv_dir, fold):
        self.csv_dir = csv_dir
        self.csv_files = {
            'train': os.path.join(csv_dir, f'train_{fold}.csv'),
            'validation': os.path.join(csv_dir, f'val_{fold}.csv'),
            'test': os.path.join(csv_dir, f'test_{fold}.csv')
        }

    def __getitem__(self, split):
        if split not in self.csv_files:
            raise ValueError(f"Invalid split name: {split}. Must be one of {list(self.csv_files.keys())}.")

        df = pd.read_csv(self.csv_files[split], dtype={'patient_id': str, 'subtype': str})
        slides = df['patient_id'].tolist()
        subtypes = df['subtype'].tolist()

        # 每个样本是 slide + subtype 标签
        dataset = [{"slide": slide, "subtype": subtype} for slide, subtype in zip(slides, subtypes)]
        return dataset


def convert_slide_id(slide_id):
    """ '0010' → 'patient_001_node_0'"""
    slide_id = f"{int(slide_id):04d}"
    patient_str = slide_id[:3]
    node_str = slide_id[3:]
    return f"patient_{patient_str}_node_{node_str}"


class PreDataset(Dataset):
    def __init__(self, datasets_list, slide_label_mapping, patient_label_mapping, h5_path):
        self.dataset = datasets_list
        self.slide_label_mapping = slide_label_mapping
        self.patient_label_mapping = patient_label_mapping
        self.h5_folder = h5_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        slide_id = entry['slide']
        slide_label = self.slide_label_mapping[entry['slide_label']]
        patient_label = self.patient_label_mapping[entry['patient_label']]

        slide_name = convert_slide_id(slide_id)
        h5_path = os.path.join(self.h5_folder, f"{slide_name}.h5")

        with h5py.File(h5_path, 'r') as f:
            features = torch.tensor(f['features'][:], dtype=torch.float32)
            coords = torch.tensor(f['coords'][:], dtype=torch.int32)

        return {
            'features': features,
            'coords': coords,
            'slide': slide_name,
            'slide_label': slide_label,
            'patient_label': patient_label
        }


class PatientDataset(Dataset):
    def __init__(self, datasets_list, patient_label_mapping, h5_path):
        self.h5_folder = h5_path
        self.patient_label_mapping = patient_label_mapping

        # 创建 patient_id → 所有 slide 列表
        self.patient_dict = {}
        for entry in datasets_list:
            pid = entry['slide'][:3]  # 提取前三位病人号，例如 '001'
            if pid not in self.patient_dict:
                self.patient_dict[pid] = []
            self.patient_dict[pid].append(entry)

        # 转为 list，方便按 index 访问
        self.patient_ids = list(self.patient_dict.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        slide_entries = self.patient_dict[patient_id]

        features_all = []
        for entry in slide_entries:
            slide_id = entry['slide']
            slide_name = convert_slide_id(slide_id)
            h5_path = os.path.join(self.h5_folder, f"{slide_name}.h5")

            with h5py.File(h5_path, 'r') as f:
                features = torch.tensor(f['embedding'][:], dtype=torch.float32)
            features_all.append(features)

        # 拼接多个 slide 的特征，例如 [n_slide, 1024]
        features_all = torch.stack(features_all, dim=0)

        # 获取病人标签（多个 entry 是同一个病人，所以任意取一个即可）
        patient_label = self.patient_label_mapping[slide_entries[0]['patient_label']]

        return {
            'patient_id': patient_id,
            'features': features_all,  # shape: [num_slide, feature_dim]
            'patient_label': patient_label
        }


class TCGASlideDataset(Dataset):
    def __init__(self, data_list, label_mapping, pt_path):
        self.dataset = data_list
        self.label_mapping = label_mapping
        self.pt_folder = pt_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        patient_id = entry['slide']
        patient_label = self.label_mapping[entry['subtype']]

        matched_files = glob.glob(os.path.join(self.pt_folder, f"{patient_id}*.pt"))

        if len(matched_files) == 0:
            raise FileNotFoundError(f"No .pt file found for patient_id: {patient_id}")
        elif len(matched_files) > 1:
            print(f"[Warning] Lots of {matched_files} being found, using the first one")

        pt_file_path = matched_files[0]
        data = torch.load(pt_file_path, map_location='cpu')

        return {
            'features': data,
            'patient_id': patient_id,
            'patient_label': patient_label
        }
