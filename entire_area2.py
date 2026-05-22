# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
from SSLD import model_srnn
from SSLD import inference_network
from SSLD import initialization
from SSLD import train
from SSLD import datapre
import copy
import yaml
import argparse
from SSLD import share
import warnings
warnings.filterwarnings('ignore', message='enable_nested_tensor is True')


save_name = 'area2_entire'


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--jobid", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--alp1", type=float, default=None)
    p.add_argument("--alp2", type=float, default=None)
    p.add_argument("--alp3", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--ini_epochs", type=int, default=None)
    p.add_argument("--num_tv", type=int, default=None)
    p.add_argument("--hidden_shape", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.jobid is not None:
        cfg["experiment"]["jobid"] = args.jobid
    if args.fold is not None:
        cfg["experiment"]["fold"] = args.fold
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.ini_epochs is not None:
        cfg["train"]["ini_epochs"] = args.ini_epochs
    if args.num_tv is not None:
        cfg["model"]["num_tv"] = args.num_tv
    if args.hidden_shape is not None:
        cfg["model"]["hidden_shape"] = args.hidden_shape
    if args.alp1 is not None:
        cfg["train"]["alp1"] = args.alp1
    if args.alp2 is not None:
        cfg["train"]["alp2"] = args.alp2
    if args.alp3 is not None:
        cfg["train"]["alp3"] = args.alp3

    # Seeds
    seed = int(cfg["experiment"].get("seed", 131))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device_cfg = cfg["system"].get("device", "auto")
    if device_cfg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_cfg)

    # dtype
    dtype_str = cfg["system"].get("dtype", "float32")
    dtype = torch.float32 if dtype_str == "float32" else torch.float64

    save_folder = cfg["experiment"]["save_dir"]
    save_fold = cfg["experiment"]["fold"]
    save_ckp = cfg["experiment"]["ckp"]


    print(f"  Data Process for Entire Set  ".center(55, '='))

    (neural_train, neural_test,
         handvel_train, handvel_test,hand_train, hand_test,force_train, force_test,
         labels_train_, labels_test_) = datapre.process(
        act_path     = "./data/act_save_single_200.npy",
        handvel_path = "./data/handvel_save_single_200.npy",
        hand_path = "./data/hand_save_single_200.npy",
        force_path = "./data/force_save_single_200.npy",
        fold=int(save_fold),
        split_point  = 100,
    )

    neural_train_, neural_test_, sca1 = datapre.normalize_train_test(neural_train, neural_test)
    handvel_train_, handvel_test_, sca2 = datapre.normalize_train_test(handvel_train, handvel_test)
    hand_train_, hand_test_, sca3 = datapre.normalize_train_test(hand_train, hand_test)
    force_train_, force_test_, sca4 = datapre.normalize_train_test(force_train, force_test)

    handvel_train=np.concatenate((handvel_train_,handvel_test_),axis=0)
    hand_train=np.concatenate((hand_train_,hand_test_),axis=0)
    force_train=np.concatenate((force_train_,force_test_),axis=0)
    neural_train=np.concatenate((neural_train_,neural_test_),axis=0)
    labels_train=np.concatenate((labels_train_,labels_test_),axis=0)

    handvel_test=handvel_train
    hand_test=hand_train
    force_test=force_train
    neural_test=neural_train
    labels_test=labels_train


    behavior_train=np.concatenate((handvel_train,hand_train,force_train),axis=-1)
    behavior_test=np.concatenate((handvel_test,hand_test,force_test),axis=-1)


    print("Final output shapes:")
    print(f"  neural_train   : {neural_train.shape}   (trials, time, neurons)")
    print(f"  neural_test    : {neural_test.shape}    (trials, time, neurons)")
    print(f"  behavior_train : {behavior_train.shape}  (trials, time, xy)")
    print(f"  behavior_test  : {behavior_test.shape}   (trials, time, xy)")
    print(f"  labels_train   : {labels_train.shape}  (trials, time, 1)")
    print(f"  labels_test    : {labels_test.shape}   (trials, time, 1)")

    # Sanity checks
    assert np.all(labels_train[:, :100, 0] == 0)
    assert np.all(labels_train[:, 100:, 0] == 1)
    assert np.all(labels_test[:,  :100, 0] == 0)
    assert np.all(labels_test[:,  100:, 0] == 1)
    print("\nOK: Label check passed.")
    print( '-'*55)
    print(f"  Start Run for Entire Set")

    jobid = int(cfg["experiment"]["jobid"])

    y_train = torch.tensor(neural_train,    dtype=dtype, device=device)
    y_test  = torch.tensor(neural_test,     dtype=dtype, device=device)
    X_train = torch.zeros(neural_train.shape,  dtype=dtype, device=device)
    X_test  = torch.zeros(neural_test.shape,   dtype=dtype, device=device)

    beh_train   = torch.tensor(behavior_train,  dtype=dtype, device=device)
    beh_test    = torch.tensor(behavior_test,   dtype=dtype, device=device)
    label_train = torch.tensor(labels_train,    dtype=dtype, device=device)
    label_test  = torch.tensor(labels_test,     dtype=dtype, device=device)

    beh_all_train = None
    beh_all_test  = None

    input_shape         = X_train.shape[2]
    beh_shape           = beh_train.shape[2]
    num_tv              = int(cfg["model"]["num_tv"])
    hidden_shape        = int(cfg["model"]["hidden_shape"])
    bottleneck_shape    = cfg["model"]["bottleneck_shape"]
    beh_private_shape   = cfg["model"]["beh_private_shape"]
    neural_private_shape = cfg["model"]["neural_private_shape"]

    ini_epochs      = int(cfg["train"]["ini_epochs"])
    epochs          = int(cfg["train"]["epochs"])
    lr              = float(cfg["train"]["lr"])
    coef_cross      = float(cfg["train"]["coef_cross"])
    coef_cross_beh  = float(cfg["train"]["coef_cross_beh"])
    batch_size      = int(cfg["train"]["batch_size"])
    init_method     = cfg["train"].get("init_method", "random")
    t_load          = cfg["train"].get("t_load", None)
    alp1            = float(cfg["train"]["alp1"])
    alp2            = float(cfg["train"]["alp2"])
    alp3            = float(cfg["train"]["alp3"])

    model        = model_srnn.Model(input_shape, beh_shape, num_tv, hidden_shape, beh_private_shape, neural_private_shape).to(device)
    rnninfer     = inference_network.RNNInfer(input_shape, hidden_shape).to(device)
    behinfer     = inference_network.BEHInfer(beh_shape, hidden_shape).to(device)
    sharedecoder = share.Decoder(hidden_shape, bottleneck_shape, hidden_shape, beh_private_shape, neural_private_shape).to(device)

    optimizer       = torch.optim.Adam(list(model.parameters()),        lr=lr)
    optimizer_rnn   = torch.optim.Adam(list(rnninfer.parameters()),     lr=lr)
    optimizer_beh   = torch.optim.Adam(list(behinfer.parameters()),     lr=lr)
    optimizer_share = torch.optim.Adam(list(sharedecoder.parameters()), lr=lr)

    scheduler       = torch.optim.lr_scheduler.StepLR(optimizer,       step_size=2000, gamma=0.8)
    scheduler_rnn   = torch.optim.lr_scheduler.StepLR(optimizer_rnn,   step_size=2000, gamma=0.8)
    scheduler_beh   = torch.optim.lr_scheduler.StepLR(optimizer_beh,   step_size=2000, gamma=0.8)
    scheduler_share = torch.optim.lr_scheduler.StepLR(optimizer_share, step_size=2000, gamma=0.8)

    (model_ini, rnninfer_ini, behinfer_ini, sharedecoder_ini,
     mse_all_ini, error_all_ini, mse_all_test_ini, error_all_test_ini,
     loss_all_ini, pos_test_all_ini) = initialization.run(
        model, rnninfer, behinfer, sharedecoder,
        optimizer, optimizer_rnn, optimizer_beh, optimizer_share,
        scheduler, scheduler_rnn, scheduler_beh, scheduler_share,
        X_train, y_train, beh_train, label_train,
        X_test, y_test, beh_test, label_test,
        beh_all_train, beh_all_test,
        num_tv, coef_cross_beh, ini_epochs, batch_size,
        alp1, alp2, alp3,
        save_name, save_folder, save_fold, save_ckp, device,
    )

    model_ini_save        = copy.deepcopy(model_ini).state_dict()
    rnninfer_ini_save     = copy.deepcopy(rnninfer_ini).state_dict()
    behinfer_ini_save     = copy.deepcopy(behinfer_ini).state_dict()
    sharedecoder_ini_save = copy.deepcopy(sharedecoder_ini).state_dict()

    (model_trained, rnninfer_trained, behinfer_trained, sharedecoder_trained,
     mse_all_train, error_all_train, mse_all_test, error_all_test,
     loss_all, pos_test_all) = train.train_(
        model_ini, rnninfer_ini, behinfer_ini, sharedecoder_ini,
        optimizer, optimizer_rnn, optimizer_beh, optimizer_share,
        scheduler, scheduler_rnn, scheduler_beh, scheduler_share,
        X_train, y_train, beh_train, label_train,
        X_test, y_test, beh_test, label_test,
        beh_all_train, beh_all_test,
        num_tv, coef_cross_beh, epochs, batch_size,
        alp1, alp2, alp3,
        save_name, save_folder, save_fold, save_ckp, device,
    )

    out_path = os.path.join(
        save_folder,
        f"{save_name}_model_hidden{hidden_shape}_{jobid}_{int(alp1)}_fold{save_fold}.pt"
    )

    torch.save({
        'num_tv':               num_tv,
        'lr':                   lr,
        'alp1':                 alp1,
        'alp2':                 alp2,
        'alp3':                 alp3,
        'bottleneck_shape':     bottleneck_shape,
        'beh_private_shape':    beh_private_shape,
        'neural_private_shape': neural_private_shape,
        'coef_cross':           coef_cross,
        'coef_cross_beh':       coef_cross_beh,
        'init_method':          init_method,
        'hidden_shape':         hidden_shape,
        'y_train':              y_train.cpu().detach().numpy(),
        'X_train':              X_train.cpu().detach().numpy(),
        'beh_train':            beh_train.cpu().detach().numpy(),
        'label_train':          label_train.cpu().detach().numpy(),
        'y_test':               y_test.cpu().detach().numpy(),
        'X_test':               X_test.cpu().detach().numpy(),
        'beh_test':             beh_test.cpu().detach().numpy(),
        'label_test':           label_test.cpu().detach().numpy(),
        'model_state_dict_ini':       model_ini_save,
        'rnninfer_state_dict_ini':    rnninfer_ini_save,
        'behinfer_state_dict_ini':    behinfer_ini_save,
        'sharedecoder_state_dict_ini': sharedecoder_ini_save,
        'optimizer_state_dict_ini':   optimizer.state_dict(),
        'model_state_dict':           model_trained.state_dict(),
        'rnninfer_state_dict':         rnninfer_trained.state_dict(),
        'behinfer_state_dict':         behinfer_trained.state_dict(),
        'sharedecoder_state_dict':     sharedecoder_trained.state_dict(),
        'optimizer_state_dict':        optimizer.state_dict(),
        'mse_all':       mse_all_train,
        'error_all':     error_all_train,
        'mse_all_test':  mse_all_test,
        'error_all_test': error_all_test,
        'loss_train':    loss_all,
        'pos_test_all':  pos_test_all,
    }, out_path)
    print(f"  Done for Entire Set ".center(55, '='))


if __name__ == "__main__":
    main()
