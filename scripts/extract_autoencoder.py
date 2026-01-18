import torch
from omegaconf import OmegaConf
from scripts.inference_test_bench import load_model_from_config

config = OmegaConf.load("configs/mobi_nusc_all-classes_512.yaml")
model = load_model_from_config(config, "checkpoints/model.ckpt")

state_dict = model.first_stage_model.state_dict()
torch.save({"state_dict": state_dict}, "checkpoints/image_vae.ckpt")