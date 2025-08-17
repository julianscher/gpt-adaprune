import argparse

import torch
from torch import nn

from src.model import TransformerModel, PretrainedModel
from src.model.mask_model import ModelMasker


def build_model(args: argparse.Namespace) -> nn.Module:
    model_args = args.model
    if model_args["family"] == "gpt2":
        model = TransformerModel(
            n_dims=model_args["n_dims"],
            n_positions=model_args["n_positions"],
            n_embd=model_args["n_embd"],
            n_layer=model_args["n_layer"],
            n_head=model_args["n_head"],
        )
    elif model_args["family"] == "pretrained-gpt2":
        model = PretrainedModel(
            n_dims=model_args["n_dims"],
            n_positions=model_args["n_positions"],
            n_embd=model_args["n_embd"],
        )
    else:
        raise NotImplementedError

    # Load pretrained model
    save_path = model_args.get("save_path", None)
    if save_path is not None:
        if '/scratch' in save_path:
            save_path = save_path.replace('/scratch', '/compute/locus-0-37')
        state = torch.load(save_path, map_location=args.device)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        if model_args["linear_probe"]:
            for param in model.parameters():
                param.requires_grad = False
            model._read_out.weight.requires_grad = True
            model._read_out.bias.requires_grad = True

    # Mask model for fine-pruning
    if args.trainer in ["adaprune_trainer"]:
        model = ModelMasker(model, args.device, float(args.weights_lr), float(args.mask_lr), float(args.mask_decay))

    return model