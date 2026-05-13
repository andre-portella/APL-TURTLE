import torch
import numpy as np
from scipy.spatial.distance import cdist
import sys


class Selector:
    def __init__(self, task_phi, device, U_index):
        self.task_phi = task_phi
        self.device = device
        self.U_index = U_index

    def select(self, strategy, n_query, features):
        """
        strategy: str
            'centroide', 'entropia', 'confianca', 'margem', 'margem_confianca'
        """

        strategies = {
            "centroide": self._select_via_centroide,
            "entropia": self._select_via_entropia,
            "confianca": self._select_via_confianca,
            "margem": self._select_via_margin_sampling,
            "margem_confianca": self._select_via_margin_sampling_confianca,
        }

        if strategy not in strategies:
            raise ValueError(f"Strategy '{strategy}' não suportada")

        return strategies[strategy](n_query, features)

    # =========================================================
    # CENTROIDE
    # =========================================================
    def _select_via_centroide(self, n_query, features):
        self.task_phi.eval()

        with torch.no_grad():
            inputs = features.to(self.device).to(torch.float32)
            logits = self.task_phi(inputs)

            cluster_assignments = torch.argmax(logits, dim=1).cpu().numpy()
            features_np = inputs.cpu().numpy()

        uniques = np.unique(cluster_assignments)
        selected_indices = []

        samples_per_cluster = max(1, n_query // len(uniques))

        for c in uniques:
            c_indices = np.where(cluster_assignments == c)[0]
            if len(c_indices) == 0:
                continue

            cluster_samples = features_np[c_indices]

            centroid = np.mean(cluster_samples, axis=0).reshape(1, -1)

            # normalização para distância cosseno
            cluster_samples_norm = cluster_samples / np.linalg.norm(cluster_samples, axis=1, keepdims=True)
            centroid_norm = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)

            dists = cdist(cluster_samples_norm, centroid_norm, metric="cosine").flatten()

            sort_idx = np.argsort(dists)[:samples_per_cluster]

            for s in sort_idx:
                global_idx = c_indices[s]
                selected_indices.append(self.U_index[int(global_idx)])

                if len(selected_indices) >= n_query:
                    break

        return selected_indices, cluster_assignments


    # =========================================================
    # ENTROPIA
    # =========================================================
    def _select_via_entropia(self, n_query, features):
        self.task_phi.eval()

        with torch.no_grad():
            inputs = features.to(self.device).to(torch.float32)
            logits = self.task_phi(inputs)

            probs = torch.softmax(logits, dim=1)
            pseudolabels = torch.argmax(logits, dim=1).cpu().numpy()

        pseudolabels_unique = np.unique(pseudolabels)
        samples_per_cluster = max(1, n_query // len(pseudolabels_unique))

        selected_indices = []

        for c in pseudolabels_unique:
            c_indices = np.where(pseudolabels == c)[0]
            if len(c_indices) == 0:
                continue

            probs_c = probs[c_indices]

            entropies = -(probs_c * torch.log(torch.clamp(probs_c, min=sys.float_info.epsilon))).sum(dim=1)

            sorted_idx = torch.argsort(entropies).cpu().numpy()
            best_local = c_indices[sorted_idx[:samples_per_cluster]]

            for idx in best_local:
                selected_indices.append(self.U_index[int(idx)])

                if len(selected_indices) >= n_query:
                    break

        return selected_indices, pseudolabels

    # =========================================================
    # CONFIANÇA
    # =========================================================
    def _select_via_confianca(self, n_query, features):
        self.task_phi.eval()

        with torch.no_grad():
            inputs = features.to(self.device).to(torch.float32)
            logits = self.task_phi(inputs)

            probs = torch.softmax(logits, dim=1)
            pseudolabels = torch.argmax(logits, dim=1).cpu().numpy()

        pseudolabels_unique = np.unique(pseudolabels)
        selected_indices = []

        for c in pseudolabels_unique:
            c_indices = np.where(pseudolabels == c)[0]
            if len(c_indices) == 0:
                continue

            probs_c = probs[c_indices]

            class_probs = probs_c[:, c]
            best_idx_local = torch.argmax(class_probs).item()

            best_sample_idx = c_indices[best_idx_local]
            selected_indices.append(self.U_index[int(best_sample_idx)])

        return selected_indices, pseudolabels

    # =========================================================
    # MARGIN SAMPLING
    # =========================================================
    def _select_via_margin_sampling(self, n_query, features):
        self.task_phi.eval()

        with torch.no_grad():
            inputs = features.to(self.device).to(torch.float32)
            logits = self.task_phi(inputs)

            probs = torch.softmax(logits, dim=1)
            pseudolabels = torch.argmax(logits, dim=1).cpu().numpy()

        pseudolabels_unique = np.unique(pseudolabels)
        selected_indices = []

        for c in pseudolabels_unique:
            c_indices = np.where(pseudolabels == c)[0]
            if len(c_indices) == 0:
                continue

            probs_c = probs[c_indices]

            top2 = probs_c.topk(2, dim=1).values
            margin_scores = top2[:, 0] - top2[:, 1]

            best_idx_local = torch.argmax(margin_scores).item()
            best_sample_idx = c_indices[best_idx_local]

            selected_indices.append(self.U_index[int(best_sample_idx)])

        return selected_indices, pseudolabels

    # =========================================================
    # MARGIN + CONFIANÇA
    # =========================================================
    def _select_via_margin_sampling_confianca(self, n_query, features):
        self.task_phi.eval()

        with torch.no_grad():
            inputs = features.to(self.device).to(torch.float32)
            logits = self.task_phi(inputs)

            probs = torch.softmax(logits, dim=1)
            pseudolabels = torch.argmax(logits, dim=1).cpu().numpy()

        pseudolabels_unique = np.unique(pseudolabels)
        selected_indices = []

        for c in pseudolabels_unique:
            c_indices = np.where(pseudolabels == c)[0]
            if len(c_indices) == 0:
                continue

            probs_c = probs[c_indices]

            top2 = probs_c.topk(2, dim=1).values
            margin_scores = top2[:, 0] - top2[:, 1]

            confidence = top2[:, 0]
            combined_score = margin_scores * confidence

            best_idx_local = torch.argmax(combined_score).item()
            best_sample_idx = c_indices[best_idx_local]

            selected_indices.append(self.U_index[int(best_sample_idx)])

        return selected_indices, pseudolabels