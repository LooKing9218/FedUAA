import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import DRDataset
from sklearn import metrics
from models import ResNetBackbone
import argparse
import time
import torchvision.transforms as transforms
import random
import tqdm
import torch.nn.functional as F
import csv

# loss function
def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)



def train(model, data_loader, optimizer, loss_fun, device, dataset, num_classes, epoch):
    print("Training on the dataset of: {} ......".format(dataset))
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    tbar = tqdm.tqdm(data_loader, desc='\r')

    u_list = []
    u_label_list = []
    for idx, data_list in enumerate(tbar):
        data, target = data_list
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)

        evidences = [F.softplus(output)]
        loss_un = 0
        alpha = dict()
        alpha[0] = evidences[0] + 1

        S = torch.sum(alpha[0], dim=1, keepdim=True)
        E = alpha[0] - 1
        b = E / (S.expand(E.shape))
        u = num_classes / S

        loss_un += ce_loss(target, alpha[0], num_classes, epoch, args.iters, device)
        loss_ACE = torch.mean(loss_un)

        loss_temp = loss_fun(b / 0.05, target)
        loss = loss_ACE+loss_temp

        loss_all += loss.item()
        loss.backward()
        total += target.size(0)
        pred = b.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
        optimizer.step()

        data_bach = b.size(0)
        un_gt = 1 - torch.eq(b.argmax(dim=-1), target).float()
        for idx in range(data_bach):
            u_list.append(u.detach().cpu()[idx].numpy())
            u_label_list.append(un_gt.cpu()[idx].numpy())

    return loss_all / len(data_loader), correct/total, u_list, u_label_list




def test(model, data_loader, loss_fun, device,num_classes,epoch):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    label_list = []
    outputs = []
    tbar = tqdm.tqdm(data_loader, desc='\r')
    with torch.no_grad():
        for step, data_list in enumerate(tbar):
            data, target = data_list
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            evidences = [F.softplus(output)]
            loss_un = 0
            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            loss_un += ce_loss(target, alpha[0], num_classes, epoch, args.iters, device)
            loss_ACE = torch.mean(loss_un)
            loss_temp = loss_fun(b/0.05, target)
            loss = loss_temp+loss_ACE
            loss_all+=loss.item()


            total += target.size(0)
            pred = b.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

            if len(output.shape) < 2:
                output = torch.unsqueeze(output, dim=0)
            out_auc = output.softmax(1).cpu().detach().float().numpy()

            one_hot = torch.zeros(data.size(0), num_classes).to(device).scatter_(1, target.unsqueeze(1), 1)
            for idx in range(data.size(0)):
                outputs.append(out_auc[idx])
                label_list.append(one_hot.cpu().detach().float().numpy()[idx])

        epoch_auc = metrics.roc_auc_score(label_list, outputs)

        return loss_all / len(data_loader), correct/total, epoch_auc


def communication(server_model, models, client_weights):
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            if 'bn' not in key and 'classifier' not in key:
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models

def prepare_data(args):
    data_base_path = './Dataset/FedDR'
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    # APTOS2019
    APTOS2019_trainset = DRDataset(data_base_path, 'APTOS2019/train_FEDAPTOS2019.csv', transform=transform_train)
    APTOS2019_valset = DRDataset(data_base_path, 'APTOS2019/valid_FEDAPTOS2019.csv', transform=transform_test)
    APTOS2019_testset = DRDataset(data_base_path, 'APTOS2019/test_FEDAPTOS2019.csv', transform=transform_test)

    # DDR
    DDR_trainset = DRDataset(data_base_path, 'DDR/train_FEDDDR.csv', transform=transform_train)
    DDR_valset = DRDataset(data_base_path, 'DDR/valid_FEDDDR.csv', transform=transform_test)
    DDR_testset = DRDataset(data_base_path, 'DDR/test_FEDDDR.csv', transform=transform_test)

    # DRR
    DRR_trainset = DRDataset(data_base_path, 'DRR/DR_Txt/train_FEDDRR.csv', transform=transform_train)
    DRR_valset = DRDataset(data_base_path, 'DRR/DR_Txt/val_FEDDRR.csv', transform=transform_test)
    DRR_testset = DRDataset(data_base_path, 'DRR/DR_Txt/test_FEDDRR.csv', transform=transform_test)

    # Messidor
    Messidor_trainset = DRDataset(data_base_path, 'Messidor/train_FEDDRMessidor.csv', transform=transform_train)
    Messidor_valset = DRDataset(data_base_path, 'Messidor/val_FEDDRMessidor.csv', transform=transform_test)
    Messidor_testset = DRDataset(data_base_path, 'Messidor/test_FEDDRMessidor.csv', transform=transform_test)

    # IDRiD
    IDRiD_trainset = DRDataset(data_base_path, 'IDRiD/train_FEDDRIDRiD.csv', transform=transform_train)
    IDRiD_valset = DRDataset(data_base_path, 'IDRiD/val_FEDDRIDRiD.csv', transform=transform_test)
    IDRiD_testset = DRDataset(data_base_path, 'IDRiD/test_FEDDRIDRiD.csv', transform=transform_test)

    APTOS2019_train_loader = torch.utils.data.DataLoader(APTOS2019_trainset, batch_size=args.batch, shuffle=True,drop_last=True)
    APTOS2019_val_loader = torch.utils.data.DataLoader(APTOS2019_valset, batch_size=args.batch, shuffle=False)
    APTOS2019_test_loader = torch.utils.data.DataLoader(APTOS2019_testset, batch_size=args.batch, shuffle=False)

    DDR_train_loader = torch.utils.data.DataLoader(DDR_trainset, batch_size=args.batch, shuffle=True,drop_last=True)
    DDR_val_loader = torch.utils.data.DataLoader(DDR_valset, batch_size=args.batch, shuffle=False)
    DDR_test_loader = torch.utils.data.DataLoader(DDR_testset, batch_size=args.batch, shuffle=False)

    DRR_train_loader = torch.utils.data.DataLoader(DRR_trainset, batch_size=args.batch, shuffle=True,drop_last=True)
    DRR_val_loader = torch.utils.data.DataLoader(DRR_valset, batch_size=args.batch, shuffle=False)
    DRR_test_loader = torch.utils.data.DataLoader(DRR_testset, batch_size=args.batch, shuffle=False)

    Messidor_train_loader = torch.utils.data.DataLoader(Messidor_trainset, batch_size=args.batch, shuffle=True,drop_last=True)
    Messidor_val_loader = torch.utils.data.DataLoader(Messidor_valset, batch_size=args.batch, shuffle=False)
    Messidor_test_loader = torch.utils.data.DataLoader(Messidor_testset, batch_size=args.batch, shuffle=False)

    IDRiD_train_loader = torch.utils.data.DataLoader(IDRiD_trainset, batch_size=args.batch, shuffle=True,drop_last=True)
    IDRiD_val_loader = torch.utils.data.DataLoader(IDRiD_valset, batch_size=args.batch, shuffle=False)
    IDRiD_test_loader = torch.utils.data.DataLoader(IDRiD_testset, batch_size=args.batch, shuffle=False)



    train_loaders = [APTOS2019_train_loader, DDR_train_loader, DRR_train_loader, Messidor_train_loader, IDRiD_train_loader]
    val_loaders = [APTOS2019_val_loader, DDR_val_loader, DRR_val_loader, Messidor_val_loader, IDRiD_val_loader]
    test_loaders = [APTOS2019_test_loader, DDR_test_loader, DRR_test_loader, Messidor_test_loader, IDRiD_test_loader]
    return train_loaders, val_loaders, test_loaders


if __name__ == '__main__':
    torch.set_num_threads(1)
    seed=1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=True, help='whether to log')
    parser.add_argument('--log_path', default='./log_path', help='whether to log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default = 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--gpu', type = int, default=0, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedUAA')
    parser.add_argument('--save_path', type = str, default='./ModelSaved/FedUAA', help='path to save the checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu))
    args.device = device


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
    log = args.log
    log_path = args.log_path

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
    logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logfile.write('===Setting===\n')
    logfile.write('    lr: {}\n'.format(args.lr))
    logfile.write('    batch: {}\n'.format(args.batch))
    logfile.write('    iters: {}\n'.format(args.iters))
    logfile.flush()

    train_loaders, val_loaders, test_loaders = prepare_data(args)
    server_model = ResNetBackbone(num_classes=5).to(device)
    loss_fun = nn.CrossEntropyLoss()
    datasets = ['APTOS2019', 'DDR', 'DRR', 'Messidor', 'IDRiD']

    class_dict = {
        'APTOS2019':5,
        'DDR':6,
        'DRR':5,
        'Messidor':4,
        'IDRiD':5
    }
    Headers = ['APTOS2019', 'DDR', 'DRR', 'Messidor', 'IDRiD']

    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    models=[
        ResNetBackbone(num_classes=class_dict['{}'.format(DS)]).to(device)  for DS in datasets
    ]
    best_changed = False


    best_epoch = 0
    best_acc = [0. for j in range(client_num)]
    best_auc = [0. for j in range(client_num)]
    start_iter = 1
    args.iters = args.iters + start_iter
    train_loss_total = []
    weights_values = []
    threshold_values = []

    for a_iter in range(start_iter, args.iters):
        u_thres = []
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))
            logfile.flush()
            loss_train = []
            for client_idx, model in enumerate(models):
                train_loss, train_acc, u_list, u_label_list = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device,
                                              dataset=datasets[client_idx],num_classes=class_dict[datasets[client_idx]],
                                                  epoch=a_iter)
                fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
                max_j = max(zip(fpr_Pri, tpr_Pri), key=lambda x: x[1] - x[0])
                try:
                    pred_thresh = thresh[list(zip(fpr_Pri, tpr_Pri)).index(max_j)]
                except:
                    pred_thresh = 0.5
                loss_train.append(train_loss)
                if wi == args.wk_iters - 1:
                    u_thres.append(pred_thresh)
            if wi == args.wk_iters-1:
                train_loss_total.append(loss_train)

        with torch.no_grad():
            client_weights = F.softmax(torch.from_numpy(np.array(u_thres)),dim=0).numpy()
            threshold_values.append(u_thres)
            weights_values.append(list(client_weights))

            server_model, models = communication(server_model, models, client_weights)

            val_acc_list = [None for j in range(client_num)]
            val_auc_list = [None for j in range(client_num)]
            for client_idx_val, model in enumerate(models):
                class_dataset = datasets[client_idx_val]
                val_loss, val_acc, val_auc = test(model, val_loaders[client_idx_val], loss_fun, device, num_classes=class_dict[class_dataset],epoch=a_iter)
                val_acc_list[client_idx_val] = val_acc
                val_auc_list[client_idx_val] = val_auc
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f} | Val  AUC: {:.4f}'.format(datasets[client_idx_val], val_loss, val_acc, val_auc), flush=True)
                logfile.write(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f} | Val  AUC: {:.4f}\n'.format(datasets[client_idx_val], val_loss, val_acc, val_auc))
                logfile.flush()


            if np.mean(val_auc_list) > np.mean(best_auc):
                for client_idx_val_1 in range(client_num):
                    best_acc[client_idx_val_1] = val_acc_list[client_idx_val_1]
                    best_auc[client_idx_val_1] = val_auc_list[client_idx_val_1]
                    best_epoch = a_iter
                    best_changed=True
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f} | Val AUC: {:.4f}'.format(datasets[client_idx_val_1], best_epoch, best_acc[client_idx_val_1], best_auc[client_idx_val_1]))
                    logfile.write(' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f} | Val AUC: {:.4f}\n'.format(datasets[client_idx_val_1], best_epoch, best_acc[client_idx_val_1], best_auc[client_idx_val_1]))
                    logfile.flush()

            if best_changed:
                print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                logfile.flush()

                torch.save({
                    'model_APTOS2019': models[0].state_dict(),
                    'model_DDR': models[1].state_dict(),
                    'model_DRR': models[2].state_dict(),
                    'model_Messidor': models[3].state_dict(),
                    'model_IDRiD': models[4].state_dict(),
                    'server_model': server_model.state_dict(),
                    'best_epoch': best_epoch,
                    'best_acc': best_acc,
                    'best_auc': best_auc,
                    'a_iter': a_iter
                }, SAVE_PATH)
                best_changed = False
                for client_idx_test, datasite in enumerate(datasets):
                    class_dataset = datasets[client_idx_test]
                    _, test_acc, test_auc = test(models[client_idx_test], test_loaders[client_idx_test], loss_fun, device, num_classes=class_dict[class_dataset],epoch=a_iter)
                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | Test AUC: {:.4f}'.format(datasite, best_epoch, test_acc, test_auc))
                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | Test AUC: {:.4f}\n'.format(datasite, best_epoch, test_acc, test_auc))
                    logfile.flush()

                logfile.flush()
        logfile.flush()
        logfile.close()
