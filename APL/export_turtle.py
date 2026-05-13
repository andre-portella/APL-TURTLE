import os
import os.path as osp
import shutil
import torch
import numpy as np
from tqdm import tqdm
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms.transforms import build_transform

def prepare_turtle_data(cfg, model, train_loader, train_features, train_labels, test_loader, dataset_loader):
    """
    Extrai representações e organiza imagens para o TURTLE.
    Encapsulado para ser chamado de dentro de qualquer Trainer do Dassl.
    """

    print(f"--- [TURTLE] Preparando dados para {cfg.MODEL.BACKBONE.NAME} ---")
    if cfg.MODEL.BACKBONE.NAME == "ViT-B/16":
        backbone = "clipvitB16"
    elif cfg.MODEL.BACKBONE.NAME== "ViT-B/32":
        backbone = "clipvitB32"

    print(f"--- [TURTLE] Preparando dados para {backbone} ---")
    # 1. Configuração de Caminhos
    representations_dir = f"/storage/andreportella/turtle/data/representations/{backbone}/"
    labels_dir = "/storage/andreportella/turtle/data/labels"
    base_turtle_dir = "/storage/andreportella/turtle/data/raw_datasets"
    dataset_name = model.cfg.DATASET.NAME

    print(f"--- [TURTLE] Iniciando preparação de dados para {dataset_name} ---")
    for d in [representations_dir, labels_dir]:
        os.makedirs(d, exist_ok=True)

    # model = model.clip_model if hasattr(model, 'clip_model') else model.model
    # model.eval()


    # 3. Função Interna de Extração
    def extract(loader, desc):
        img_feats, img_feats_normalized, dot_feats, dot_feats_normalized, labels,  = [], [], [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extraindo features"):
                image = batch["img"].cuda()
                label = batch["label"].cuda()

                preds, img_features, txt_features = model.model_inference(image, get_feature=True)

                img_features_norm = img_features / img_features.norm(dim=-1, keepdim=True)
                text_features_norm = txt_features / txt_features.norm(dim=-1, keepdim=True)
                
                # confidence
                preds = torch.nn.functional.softmax(preds, dim=1)
                preds_np = preds.cpu().numpy()

                dot_features = torch.matmul(preds, txt_features)

                img_feats.append(img_features.cpu())
                img_feats_normalized.append(img_features_norm.cpu())

                dot_feats.append(dot_features.cpu())
                labels.append(label.cpu())

                
        
        # Concatenação: [Dimensão Imagem (512) + Dimensão Dot (512)] = 1024
        combined = torch.cat([torch.cat(img_feats), torch.cat(dot_feats)], dim=1)
        return combined, torch.cat(labels).numpy(), torch.cat(img_feats).numpy()

    # Executa a extração
    train_feats, train_labels, _ = extract(train_loader, "Features Treino")
    test_feats, test_labels, _ = extract(test_loader, "Features Teste")
    # train_val_feats, train_val_labels, _ = extract(dataset_loader, "Features Dataset Completo")


    # 4. Salvamento dos Arquivos
    print("Salvando arquivos .npy...")
    np.save(osp.join(representations_dir, f"{dataset_name}_train.npy"), train_feats)
    np.save(osp.join(labels_dir, f"{dataset_name}_train.npy"), train_labels)
    np.save(osp.join(representations_dir, f"{dataset_name}_val.npy"), test_feats)
    np.save(osp.join(labels_dir, f"{dataset_name}_val.npy"), test_labels)

    # Compatibilidade com prefixos do TURTLE
    name_map = {"OxfordPets": "pets", "OxfordFlowers": "flowers", "DescribableTextures": "dtd", "CIFAR10_Custom": "cifar10"}
    if dataset_name in name_map:
        p = name_map[dataset_name]
        np.save(osp.join(representations_dir, f"{p}_train.npy"), train_feats)
        np.save(osp.join(labels_dir, f"{p}_train.npy"), train_labels)
        np.save(osp.join(representations_dir, f"{p}_val.npy"), test_feats)
        np.save(osp.join(labels_dir, f"{p}_val.npy"), test_labels)

    print(f"--- [TURTLE] Dados salvos em {representations_dir} ---")

    # return train_val_feats