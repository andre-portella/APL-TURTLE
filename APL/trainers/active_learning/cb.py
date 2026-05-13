from .AL import AL
import torch
import copy
import numpy as np

from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import entropy

import random

from .selector import Selector
from .turtle_runner import TurtleRunner
# from hume_runner import HumeRunner

import sys
import os
import sys
import os
import importlib.util

from torch.utils.data import Dataset, DataLoader
from .turtle_data import prepare_turtle_data



class CB(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, val_set, n_class, device, idx, test_loader, dataset, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.labeled_in_set = val_set
        self.device= device
        self.idx = idx
        # self.full_dataset = kwargs.get('full_dataset')
        self.test_loader = test_loader
        self.dataset = dataset
        
    def get_features(self):
        if self.idx:
            self.model.eval()
        labeled_features, unlabeled_features = None, None
        with torch.no_grad():

            print(f"size labeled_in_set: {len(self.labeled_in_set)}")
            print(f"size unlabeled_set: {len(self.unlabeled_set)}")
            labeled_in_loader = build_data_loader(
                self.cfg,
                data_source=self.labeled_in_set, 
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            labels_org, labels_pse = np.array([]), np.array([])
            scores = np.array([])
            # generate entire labeled_in features set
            for data in labeled_in_loader:
                inputs = data["img"].cuda()
                labels = data["label"].cuda()
                preds, img_features, txt_features = self.model(inputs, get_feature=True, get_text_feature=True)
                p_labels = torch.argmax(preds, dim=1)

                labels_org = np.append(labels_org, labels.cpu().numpy())
                labels_pse = np.append(labels_pse, p_labels.cpu().numpy())

                # confidence
                preds = torch.nn.functional.softmax(preds, dim=1)
                preds_np = preds.cpu().numpy()
                score = np.max(preds_np, axis=1)
                scores = np.append(scores, score)

                dot_features = torch.matmul(preds, txt_features)
                features = torch.cat([img_features, dot_features], axis=1)
                # features = (img_features + dot_features) / 2

                if labeled_features is None:
                    labeled_features = features
                else:
                    labeled_features = torch.cat((labeled_features, features), 0)

            # generate entire unlabeled features set
            for data in unlabeled_loader:
                inputs = data["img"].cuda()
                labels = data["label"]
                if self.idx:
                    preds, img_features, txt_features = self.model(inputs, get_feature=True, get_text_feature=True)
                else:
                    preds, img_features, txt_features = self.model.model_inference(inputs, get_feature=True)
                p_labels = torch.argmax(preds, dim=1)

                labels_org = np.append(labels_org, labels.cpu().numpy())
                labels_pse = np.append(labels_pse, p_labels.cpu().numpy())

                # confidence
                preds = torch.nn.functional.softmax(preds, dim=1)
                preds_np = preds.cpu().numpy()
                score = np.max(preds_np, axis=1)
                scores = np.append(scores, score)

                dot_features = torch.matmul(preds, txt_features)
                features = torch.cat([img_features, dot_features], axis=1)
                # features = (img_features + dot_features) / 2

                if unlabeled_features is None:
                    unlabeled_features = features
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)
            
        return labeled_features, unlabeled_features, scores, labels_org, labels_pse, unlabeled_loader
    
    def get_features_train_val(self):

        if self.dataset is not None:
            train_val = (
                list(self.dataset.train_x) + 
                list(self.dataset.val) + 
                # list(self.full_dataset.test) + 
                list(self.unlabeled_set)
            )
            print(f">>> Processando dataset completo: {len(train_val)} imagens")


        # # 2. Criar Loader para o bloco total
        dataset_loader = build_data_loader(
            self.cfg,
            data_source=train_val,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            tfm=build_transform(self.cfg, is_train=False),
            is_train=False,
        )
        return dataset_loader
    
    

    def save_data_for_tsne(self, features, turtle_labels, real_labels, filename):

        save_path = filename
        print(f">>> Salvando dados para t-SNE em: {save_path}")
        
        np.savez(
            save_path,
            features=features,
            turtle_labels=turtle_labels,
            real_labels=real_labels
        )


    def select(self, n_query, **kwargs):
        labeled_features, unlabeled_features, scores, labels_org, labels_pse, unlabeled_loader = self.get_features()


        # dataset_loader = self.get_features_train_val()
        dataset_loader = None

        prepare_turtle_data(self.cfg, self.model, unlabeled_loader, unlabeled_features, labels_org, self.test_loader, dataset_loader)


        self.turtle = TurtleRunner(self.cfg, self.device)

        dataset_name = self.cfg.DATASET.NAME

        self.turtle.train(dataset_name)

        checkpoint = self.turtle.load_checkpoint(dataset_name)
        dim = unlabeled_features.shape[1]
        n_class = self.n_class

        self.task_phi = self.turtle.init_task_encoder(
            dim=dim,
            n_class=n_class,
            checkpoint=checkpoint
        )

        selector = Selector(
            task_phi=self.task_phi,
            device=self.device,
            U_index=self.U_index
            # U_index=list(range(len(train_val_feats)))
        )

        # aqui definimos a estratégia de seleção
        Q_index, pseudo_labels = selector.select(
            strategy=self.cfg.STRATEGY,
            n_query=n_query,
            features=unlabeled_features
        )

        self.save_data_for_tsne(
            features=unlabeled_features.cpu().numpy(),
            turtle_labels=pseudo_labels,
            real_labels=labels_org,
            filename=f"tsne_data_{self.cfg.DATASET.NAME}.npz"
        )
        return Q_index, 