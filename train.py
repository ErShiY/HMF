import gc
import os

import h5py
import torch
import torch.multiprocessing as mp
from torch.cuda import amp
import torch.optim as optim
from dataloader import CSVReader, CSVReaderTCGA, PreDataset, PatientDataset, TCGASlideDataset
from model import Model_HMF
from utils import setup_logger, EarlyStopping
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from collections import OrderedDict
import torch.distributed as dist
from tqdm import tqdm
import random
import time
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


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ["MASTER_PORT"] = str(29500 + random.randint(1, 1000))
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_model_camelyon(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn', force=True)

    datasets_train = CSVReader(args.csv_path, args.fold)['train']
    datasets_val = CSVReader(args.csv_path, args.fold)['validation']

    train_dataset = PreDataset(datasets_train, slide_label_mapping, patient_label_mapping, args.h5_path)
    val_dataset = PreDataset(datasets_val, slide_label_mapping, patient_label_mapping, args.h5_path)

    train_patient_dataset = PatientDataset(datasets_train, patient_label_mapping, "slide_embeddings")
    val_patient_dataset = PatientDataset(datasets_val, patient_label_mapping, "slide_embeddings_val")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size_slide, sampler=train_sampler, num_workers=4,
                                  drop_last=True, pin_memory=True)

    train_patient_sampler = DistributedSampler(train_patient_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader_patient_train = DataLoader(train_patient_dataset, batch_size=args.batch_size_patient,
                                          sampler=train_patient_sampler, num_workers=4, drop_last=True, pin_memory=True)

    if rank == 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size_slide, shuffle=False, num_workers=4,
                                pin_memory=True)
        val_patient_loader = DataLoader(val_patient_dataset, batch_size=args.batch_size_patient, shuffle=False,
                                        num_workers=4,
                                        pin_memory=True)
    else:
        val_loader = None
        val_patient_loader = None

    model = Model_HMF(args.h5_path, args.weight_attention_checkpoint, args.t5model_path, args.text_data_path,
                      args.ratio)
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # checkpoint_path = "outputs/camelyon/best_model.pt"
    # state_dict = torch.load(checkpoint_path, map_location={'cuda:0': f'cuda:{rank}'})
    # ddp_model.module.load_state_dict(state_dict)
    # load_result = ddp_model.module.load_state_dict(state_dict, strict=False)
    # if rank == 0:
    #     print("✅ Load result:")
    #     print("  Missing keys:", load_result.missing_keys)
    #     print("  Unexpected keys:", load_result.unexpected_keys)

    slide_modules = ['weight_attention', 'text_encoder', 'BiCrossAttention', 'ContrastiveLoss', 'Slides_Classifier']
    patient_modules = ['Patient_Pooling', 'Patient_classifier']

    slide_params = []
    patient_params = []

    for name, module in ddp_model.module.models.items():
        if name in slide_modules:
            slide_params += list(module.parameters())
        elif name in patient_modules:
            patient_params += list(module.parameters())

    for name in patient_modules:
        for param in ddp_model.module.models[name].parameters():
            param.requires_grad = False

    optimizer_slide = torch.optim.AdamW(filter(lambda p: p.requires_grad, slide_params), lr=args.lr)
    # optimizer_patient = torch.optim.AdamW(filter(lambda p: p.requires_grad, patient_params), lr=args.lr)
    scaler = amp.GradScaler()

    if rank == 0:
        save_dir = os.path.join("outputs", args.task_target)
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "output.log")
        logger = setup_logger(log_file=log_file)
        early_stopping = EarlyStopping(patience=5, delta=0.001, save_path=os.path.join(save_dir, f"best_model_slides_{args.ratio}.pt"))

    torch.autograd.set_detect_anomaly(True)

    log_message = f"This experience training {args.task_target}"

    if args.subtyping_task:
        log_message += f", subtyping_task={args.subtyping_task}"

    log_message += f", with lr={args.lr} ratio={args.ratio}, fold={args.fold}"

    logger.info("————————————————————————————————————————————")
    logger.info(log_message)
    logger.info("____________________________________________")

    for epoch in range(args.max_epochs):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        ddp_model.train()

        pbar_train = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{args.max_epochs}", leave=True,
                          disable=(rank != 0))
        total_con_loss, total_slide_loss, total_loss = 0, 0, 0
        total_batches = 0

        for batch_idx, batch in enumerate(pbar_train):

            slide_name = batch["slide"][0]
            features = batch['features'][0].to(device, non_blocking=True)  # [N, D]
            coords = batch['coords'][0].to(device, non_blocking=True)  # [N, 2]
            slide_label = batch['slide_label'][0].to(device, non_blocking=True)  # int
            patient_label = batch['patient_label'][0].to(device, non_blocking=True)  # int

            slide_result = ddp_model.module.slide_function(features, coords, slide_label, patient_label)

            con_loss = slide_result['con_loss']
            slide_loss = slide_result['slide_loss']
            slide_embedding = slide_result['slide_embedding']

            os.makedirs("slide_embeddings", exist_ok=True)
            save_path = os.path.join("slide_embeddings", f"{slide_name}.h5")
            if isinstance(slide_embedding, torch.Tensor):
                slide_embedding = slide_embedding.detach().cpu().numpy()
            with h5py.File(save_path, "w") as hf:
                hf.create_dataset("embedding", data=slide_embedding)

            # Combine losses for the current batch
            total_con_loss += con_loss
            total_slide_loss += slide_loss
            batch_loss = con_loss + slide_loss
            total_loss += batch_loss

            total_batches += 1

            scaler.scale(batch_loss).backward()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(pbar_train):
                scaler.step(optimizer_slide)
                scaler.update()
                optimizer_slide.zero_grad(set_to_none=True)

            del features, coords, slide_label, patient_label
            del batch_loss, con_loss, slide_loss
            torch.cuda.empty_cache()
            gc.collect()

            if total_batches > 0:
                pbar_train.set_postfix(OrderedDict([
                    ('Loss', f"{(total_loss / total_batches).item():.4f}")
                ]))

        # patient process
        total_patient_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader_patient_train, desc="Patient", leave=True)):
            features = batch['features'][0].to(device, non_blocking=True)
            patient_label = batch['patient_label'][0].to(device, non_blocking=True)
            patient_result = ddp_model.module.patient_function(features, patient_label)

            patient_loss = patient_result['patient_loss']
            total_patient_loss += patient_loss

            # scaler.scale(patient_loss).backward()
            # scaler.step(optimizer_patient)
            # scaler.update()
            # optimizer_patient.zero_grad(set_to_none=True)

            del features, patient_label, patient_result
            del patient_loss
            torch.cuda.empty_cache()
            gc.collect()

        epoch_time = time.time() - start_time

        if rank == 0:
            avg_train_loss = total_loss / total_batches if total_loss > 0 else 0
            avg_con_loss = total_con_loss / total_batches if total_con_loss > 0 else 0
            avg_slide_loss = total_slide_loss / total_batches if total_slide_loss > 0 else 0
            avg_patient_loss = total_patient_loss / total_batches if total_patient_loss > 0 else 0

            logger.info(f"Epoch {epoch + 1}/{args.max_epochs} finished in {epoch_time:.2f} seconds")
            logger.info(f"    | Batch_loss: {avg_train_loss.item():.4f} |"
                        f"ConLoss: {avg_con_loss:.2f}, Slide_loss: {avg_slide_loss:.4f}, Patient_Loss: {avg_patient_loss.item():.4f}")

        # Validation process
        if rank == 0 and (epoch + 1) % args.validate_freq == 0:
            with torch.no_grad():
                ddp_model.module.eval()
                idx = 0

                all_slide_preds = []
                all_slide_labels = []

                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                    idx += 1
                    slide_name = batch["slide"][0]
                    features = batch['features'][0].to(device, non_blocking=True)  # [N, D]
                    coords = batch['coords'][0].to(device, non_blocking=True)  # [N, 2]
                    slide_label = batch['slide_label'][0].to(device, non_blocking=True)  # int
                    patient_label = batch['patient_label'][0].to(device, non_blocking=True)  # int

                    slide_result = ddp_model.module.slide_function(features, coords, slide_label, patient_label)

                    slide_logits = slide_result['slide_logits']
                    slide_embedding = slide_result['slide_embedding']

                    os.makedirs("slide_embeddings_val", exist_ok=True)
                    save_path = os.path.join("slide_embeddings_val", f"{slide_name}.h5")
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

                logger.info(f"  Validation Metrics with Slides - "
                            f"Accuracy: {avg_acc:.4f}, Precision: {avg_pre:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}")

                idx_patient = 0
                all_patient_preds = []
                all_patient_labels = []

                for batch_idx, batch in enumerate(tqdm(val_patient_loader, desc="Patient Validation", leave=False)):
                    idx_patient += 1
                    features = batch['features'][0].to(device, non_blocking=True)
                    patient_label = batch['patient_label'][0].to(device, non_blocking=True)
                    patient_result = ddp_model.module.patient_function(features, patient_label)
                    patient_logits = patient_result['patient_logits']

                    pred_patient_label = patient_logits.argmax(-1).item()

                    all_patient_preds.append(pred_patient_label)
                    all_patient_labels.append(batch['patient_label'][0].item())

                avg_patient_acc = accuracy_score(all_patient_labels, all_patient_preds)
                avg_patient_pre = precision_score(all_patient_labels, all_patient_preds, average='macro',
                                                  zero_division=0)
                avg_patient_recall = recall_score(all_patient_labels, all_patient_preds, average='macro',
                                                  zero_division=0)
                avg_patient_f1 = f1_score(all_patient_labels, all_patient_preds, average='macro', zero_division=0)

                logger.info(f"  Validation Metrics with patients - "
                            f"Accuracy: {avg_patient_acc:.4f}, Precision: {avg_patient_pre:.4f}, Recall: {avg_patient_recall:.4f}, F1: {avg_patient_f1:.4f}")

                judgement_index = avg_acc
                # judgement_index = 0.5 * avg_acc + 0.5 * avg_patient_acc
                early_stopping(judgement_index, ddp_model.module)
                if early_stopping.early_stop:
                    logger.info("⛔ Early stopping triggered. Training stopped.")
                    break


def train_model_tcga(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn', force=True)

    datasets_train = CSVReaderTCGA(args.csv_path, args.fold)['train']
    datasets_val = CSVReaderTCGA(args.csv_path, args.fold)['validation']

    if args.subtyping_task == 'nsclc':
        train_dataset = TCGASlideDataset(datasets_train, tcga_nsclc_label_mapping, args.pt_path)
        val_dataset = TCGASlideDataset(datasets_val, tcga_nsclc_label_mapping, args.pt_path)

    elif args.subtyping_task == 'brca':
        train_dataset = TCGASlideDataset(datasets_train, tcga_brca_label_mapping, args.pt_path)
        val_dataset = TCGASlideDataset(datasets_val, tcga_brca_label_mapping, args.pt_path)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size_slide, sampler=train_sampler, num_workers=4,
                                  drop_last=True, pin_memory=True)

    if rank == 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size_slide, shuffle=False, num_workers=4,
                                pin_memory=True)
    else:
        val_loader = None

    model = Model_HMF(args.h5_path, args.weight_attention_checkpoint, args.t5model_path, args.text_data_path, args.ratio, args.subtyping_task)
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(ddp_model.module.parameters(), lr=args.lr)
    scaler = amp.GradScaler()

    if rank == 0:
        save_dir = os.path.join("outputs", args.task_target)
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "output.log")
        logger = setup_logger(log_file=log_file)
        early_stopping = EarlyStopping(patience=5, delta=0.001, save_path=os.path.join(save_dir, "best_model.pt"))

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(args.max_epochs):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        ddp_model.train()

        pbar_train = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{args.max_epochs}", leave=True,
                          disable=(rank != 0))
        total_con_loss, total_patient_loss, total_loss = 0, 0, 0
        total_batches = 0

        for batch_idx, batch in enumerate(pbar_train):

            features = batch['features'][0].to(device, non_blocking=True)  # [N, D]
            label = batch['patient_label'][0].to(device, non_blocking=True)  # int

            result = ddp_model.module.slide_only_function(features, label)

            con_loss = result['con_loss']
            patient_loss = result['patient_loss']

            # Combine losses for the current batch
            total_con_loss += con_loss
            total_patient_loss += patient_loss
            batch_loss = con_loss + patient_loss
            total_loss += batch_loss

            total_batches += 1

            scaler.scale(batch_loss).backward()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(pbar_train):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            del features, label
            del batch_loss, con_loss, patient_loss
            torch.cuda.empty_cache()
            gc.collect()

            if total_batches > 0:
                pbar_train.set_postfix(OrderedDict([
                    ('Loss', f"{(total_loss / total_batches).item():.4f}")
                ]))

            epoch_time = time.time() - start_time

        if rank == 0:
            avg_train_loss = total_loss / total_batches if total_loss > 0 else 0
            avg_con_loss = total_con_loss / total_batches if total_con_loss > 0 else 0
            avg_patient_loss = total_patient_loss / total_batches if total_patient_loss > 0 else 0

            logger.info(f"Epoch {epoch + 1}/{args.max_epochs} finished in {epoch_time:.2f} seconds")
            logger.info(f"    | Batch_loss: {avg_train_loss.item():.4f} |"
                        f"ConLoss: {avg_con_loss:.2f}, Patient_Loss: {avg_patient_loss.item():.4f}")

        # Validate process
        if rank == 0 and (epoch + 1) % args.validate_freq == 0:
            with torch.no_grad():
                ddp_model.module.eval()
                idx = 0

                all_preds = []
                all_labels = []

                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                    idx += 1

                    features = batch['features'][0].to(device, non_blocking=True)  # [N, D]
                    patient_label = batch['patient_label'][0].to(device, non_blocking=True)  # int

                    result = ddp_model.module.slide_only_function(features, patient_label)

                    logits = result['patient_logits']

                    pred_label = logits.argmax(-1).item()
                    all_preds.append(pred_label)
                    all_labels.append(batch['patient_label'][0].item())

                avg_acc = accuracy_score(all_preds, all_labels)
                avg_pre = precision_score(all_preds, all_labels, average='macro', zero_division=0)
                avg_rec = recall_score(all_preds, all_labels, average='macro', zero_division=0)
                avg_f1 = f1_score(all_preds, all_labels, average='macro', zero_division=0)

                logger.info(f"  Validation Metrics with Patients in TCGA {args.subtyping_task} - "
                            f"Accuracy: {avg_acc:.4f}, Precision: {avg_pre:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}")

                judgement_index = avg_acc
                early_stopping(judgement_index, ddp_model.module)
                if early_stopping.early_stop:
                    logger.info("⛔ Early stopping triggered. Training stopped.")
                    break
