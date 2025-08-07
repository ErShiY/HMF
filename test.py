import argparse
import os
import h5py
import torch
from torch.utils.data import DataLoader
from dataloader import CSVReader, PreDataset, PatientDataset, TCGASlideDataset, CSVReaderTCGA
from model import Model_HMF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

slide_label_mapping = {
    "negative": 0,
    "itc": 1,
    "micro": 2,
    "macro": 3
}

patient_label_mapping = {
    "pN0": 0,
    "pN0(i+)": 1,
    "pN1mi": 2,
    "pN1": 3,
    "pN2": 4
}

tcga_nsclc_label_mapping = {
    'LUAD': 0,
    'LUSC': 1
}

tcga_brca_label_mapping = {
    'IDC': 0,
    'ILC': 1
}



def test_camelyon(model, dataloader_test, dataloader_patient_test, device):
    model.eval()
    with torch.no_grad():
        all_slide_preds, all_slide_labels = [], []
        for batch in dataloader_test:
            slide_name = batch["slide"][0]
            features = batch['features'][0].to(device, non_blocking=True)
            coords = batch['coords'][0].to(device, non_blocking=True)
            slide_label = batch['slide_label'][0].to(device, non_blocking=True)
            patient_label = batch['patient_label'][0].to(device, non_blocking=True)

            slide_result = model.slide_function(features, coords, slide_label, patient_label)
            slide_logits = slide_result['slide_logits']
            slide_embedding = slide_result['slide_embedding']

            os.makedirs("slide_embeddings_test", exist_ok=True)
            save_path = os.path.join("slide_embeddings_test", f"{slide_name}.h5")
            if isinstance(slide_embedding, torch.Tensor):
                slide_embedding = slide_embedding.detach().cpu().numpy()
            with h5py.File(save_path, "w") as hf:
                hf.create_dataset("embedding", data=slide_embedding)

            pred_slide_label = slide_logits.argmax(-1).item()
            all_slide_preds.append(pred_slide_label)
            all_slide_labels.append(batch['slide_label'][0].item())

        avg_acc = accuracy_score(all_slide_labels, all_slide_preds)
        avg_pre = precision_score(all_slide_labels, all_slide_preds, average='macro', zero_division=0)
        avg_rec = recall_score(all_slide_labels, all_slide_preds, average='macro', zero_division=0)
        avg_f1 = f1_score(all_slide_labels, all_slide_preds, average='macro', zero_division=0)

        print(f"\n📊 Slide-Level Metrics:\n"
              f"  Accuracy: {avg_acc:.4f}\n"
              f"  Precision: {avg_pre:.4f}\n"
              f"  Recall: {avg_rec:.4f}\n"
              f"  F1: {avg_f1:.4f}")

        all_patient_preds, all_patient_labels = [], []
        for batch in dataloader_patient_test:
            features = batch['features'][0].to(device, non_blocking=True)
            patient_label = batch['patient_label'][0].to(device, non_blocking=True)
            patient_result = model.patient_function(features, patient_label)
            patient_logits = patient_result['patient_logits']

            pred_patient_label = patient_logits.argmax(-1).item()
            all_patient_preds.append(pred_patient_label)
            all_patient_labels.append(batch['patient_label'][0].item())

        avg_patient_acc = accuracy_score(all_patient_labels, all_patient_preds)
        avg_patient_pre = precision_score(all_patient_labels, all_patient_preds, average='macro', zero_division=0)
        avg_patient_rec = recall_score(all_patient_labels, all_patient_preds, average='macro', zero_division=0)
        avg_patient_f1 = f1_score(all_patient_labels, all_patient_preds, average='macro', zero_division=0)

        print(f"\n📊 Patient-Level Metrics:\n"
              f"  Accuracy: {avg_patient_acc:.4f}\n"
              f"  Precision: {avg_patient_pre:.4f}\n"
              f"  Recall: {avg_patient_rec:.4f}\n"
              f"  F1: {avg_patient_f1:.4f}")

def test_tcga(model, dataloader_test, device):
    model.eval()
    with torch.no_grad():

        all_preds, all_labels = [], []
        for batch in dataloader_test:

            features = batch['features'][0].to(device, non_blocking=True)  # [N, D]
            patient_label = batch['patient_label'][0].to(device, non_blocking=True)  # int

            result = model.slide_only_function(features, patient_label)

            logits = result['patient_logits']

            pred_label = logits.argmax(-1).item()
            all_preds.append(pred_label)
            all_labels.append(batch['patient_label'][0].item())

        avg_acc = accuracy_score(all_preds, all_labels)
        avg_pre = precision_score(all_preds, all_labels, average='macro', zero_division=0)
        avg_rec = recall_score(all_preds, all_labels, average='macro', zero_division=0)
        avg_f1 = f1_score(all_preds, all_labels, average='macro', zero_division=0)

        print(f"\n📊 Patient-Level Metrics: TCGA\n"
              f"  Accuracy: {avg_acc:.4f}\n"
              f"  Precision: {avg_pre:.4f}\n"
              f"  Recall: {avg_rec:.4f}\n"
              f"  F1: {avg_f1:.4f}")


def main_camelyon(args):

    checkpoint_path = f"outputs/{args.task_target}/best_model_slides_{args.ratio}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_test = CSVReader(args.csv_path, args.fold)['test']
    test_dataset = PreDataset(datasets_test, slide_label_mapping, patient_label_mapping, args.h5_path)
    test_patient_dataset = PatientDataset(datasets_test, patient_label_mapping, "slide_embeddings_test")

    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_patient_test = DataLoader(test_patient_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = Model_HMF(args.h5_path, args.weight_attention_checkpoint, args.t5model_path, args.text_data_path, args.ratio).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=False)

    print("✅ Model loaded successfully")
    print("  Missing keys:", load_result.missing_keys)
    print("  Unexpected keys:", load_result.unexpected_keys)

    test_camelyon(model, dataloader_test, dataloader_patient_test, device)

def main_tcga(args):

    checkpoint_path = f"outputs/{args.task_target}/best_model_patient_{args.ratio}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_test = CSVReaderTCGA(args.csv_path, args.fold)['test']

    if args.subtyping_task == 'nsclc':
        test_dataset = TCGASlideDataset(datasets_test, tcga_nsclc_label_mapping, args.pt_path)

    elif args.subtyping_task == 'brca':
        test_dataset = TCGASlideDataset(datasets_test, tcga_brca_label_mapping, args.pt_path)

    dataloader_test = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = Model_HMF(args.h5_path, args.weight_attention_checkpoint, args.t5model_path, args.text_data_path, args.ratio).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=False)

    print("✅ Model loaded successfully")
    print("  Missing keys:", load_result.missing_keys)
    print("  Unexpected keys:", load_result.unexpected_keys)

    test_tcga(model, dataloader_test, device)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MMMIL Model')

    # Model and data paths
    parser.add_argument('--weight_attention_checkpoint', type=str,
                        default='F:/MM_First/used_checkpoint/new_checkpoint.pth')
    parser.add_argument('--h5_path', type=str, default='E:/h5_files/')
    parser.add_argument('--pt_path', type=str, default='H:/TCGA-BRCA R50/pt_files/')
    parser.add_argument('--t5model_path', type=str, default='F:/MM_First/used_checkpoint/T5/')

    # Hyperparameters
    parser.add_argument('--task_target', type=str, default='camelyon', help='choose task (camelyon or tcga)')
    parser.add_argument('--subtyping_task', type=str, default=None, help='choose brca or nsclc')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--num_positive', type=int, default=5)
    parser.add_argument('--log_path', type=str, default='result')
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)

    return parser.parse_args()


def resolve_paths(args):
    if args.task_target == 'camelyon':
        args.csv_path = 'data_splits_stage_folds'
        args.text_data_path = 'text/camelyon17.json'

    elif args.task_target == 'tcga' and args.subtyping_task == 'nsclc':
        args.csv_path = 'data_splits_tcga_nsclc_folds'
        args.text_data_path = 'text/tcga.json'

    elif args.task_target == 'tcga' and args.subtyping_task == 'brca':
        args.csv_path = 'data_splits_tcga_brca_folds'
        args.text_data_path = 'text/tcga.json'
    else:
        raise ValueError("Unknown task_target/subtyping_task combination")

    return args


if __name__ == "__main__":
    args = parse_args()
    args = resolve_paths(args)
    if args.task_target == 'camelyon':
        main_camelyon(args)
    else:
        main_tcga(args)