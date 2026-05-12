import os
import sys
import torch
from run_turtle import run as train_turtle



class TurtleRunner:
    def __init__(self, cfg, device):
        
        self.turtle_path = "/storage/andreportella/turtle"
        self.device = device
        self.cfg = cfg

        if cfg.MODEL.BACKBONE.NAME == "ViT-B/16":
            backbone = "clipvitB16"
        elif self.cfg.MODEL.BACKBONE.NAME== "ViT-B/32":
            backbone = "clipvitB32"

        self.backbone = backbone


        if self.turtle_path not in sys.path:
            sys.path.insert(0, self.turtle_path)

        self.train_turtle = train_turtle

    def train(self, dataset_name):
        original_path = sys.path.copy()

        turtle_data = os.path.join(self.turtle_path, "data")

        seed = str(self.cfg.SEED)
        
        args_list = [
            '--dataset', dataset_name,
            '--phis', self.backbone,
            '--root_dir', turtle_data,
            '--seed', seed,
            '--warm_start',
            '--inner_lr', '0.0015',
            '--outer_lr', '0.0025',      
        ]

        print(f"\n>>> Iniciando TURTLE...")
        self.train_turtle(args_list)

        sys.path = original_path
        

    def load_checkpoint(self, dataset_name):
        checkpoint_dir = os.path.join(self.turtle_path, f"data/task_checkpoints/1space/{self.backbone}")
        checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)

        ckpt_name = f"turtle_{self.backbone}_innerlr0.0015_outerlr0.0025_T6000_M10_warmstart_gamma10.0_bs10000_seed{self.cfg.SEED}.pt"
        full_path = os.path.join(checkpoint_dir, ckpt_name)

        print(f">>> Carregando checkpoint TURTLE: {full_path}")
        return torch.load(full_path, map_location=self.device)

    def init_task_encoder(self, dim, n_class, checkpoint):
        task_phi = torch.nn.Linear(dim, n_class).to(self.device).to(torch.float32)
        task_phi.load_state_dict(checkpoint['phi1'])
        task_phi.eval()
        return task_phi