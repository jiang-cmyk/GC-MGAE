from src.model import ae
import torch
from torch import optim as optim
import torch.nn as nn
import dgl
import logging.config
from dgl.data import (
    CoraGraphDataset, 
    CiteseerGraphDataset,
    # PubmedGraphDataset
)
import argparse
from tqdm import tqdm
import numpy as np
from src.evaluation import node_classification_evaluation
import time, json, sys
GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    # "pubmed": PubmedGraphDataset,
    # "ogbn-arxiv": DglNodePropPredDataset
}
def get_logger(name):
    name += '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
    config_dir = './config/'
    log_dir = './log/'
    config_dict = json.load(open( config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-').replace(':', '-')

    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s- [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def train_epoch(epoch):
    model.train()
    loss, ce_loss = model(graph, x)
    # loss += 0.2 * ce_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    
    if (epoch + 1) % 100 == 0:
        # print(f'epoch: {epoch}, loss:', loss.item())
        tacc, elacc = node_classification_evaluation(model, graph, x, 7, 0.01, 0.0001, 300, device, True, mute=True)
        #if (epoch + 1) % 500 == 0 or True:
            # print(loss.item(), ce_loss.item())
        logger.info(f"# | Epoch={epoch:04d} # test_acc: {tacc:.4f}, # early-stopping_acc: {elacc:.4f}")
        acc_list.append(tacc)
        estp_acc_list.append(elacc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dataset_path', type=str, default="./datasets")
    parser.add_argument('--param', type=str, default='local.json')
    parser.add_argument('--seed', type=int, default=39788)
    # <<<<<<< HEAD
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument('--batch_size', type=int, default=1024)
    # =======
    # >>>>>>> e012940f83f2f427b2bcf29a82fc60c0a007227a
    parser.add_argument('--verbose', type=str, default='train,eval')
    parser.add_argument('--cls_seed', type=int, default=12345)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--cd', type=str, default='leiden')
    parser.add_argument('--ced_thr', type=float, default=1.)
    parser.add_argument('--cav_thr', type=float, default=1.)
    device = torch.device("cuda")

    logger = get_logger("Cora01")

    dataset = GRAPH_DICT["cora"]()
    graph = dataset[0].to(device)
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    x = graph.ndata["feat"]
    best_acc, best_early_stopping = [], []
    for i in range(3):
        # if True:
        # l, j, k =i % 10 *0.1, i // 10 % 10 * 0.1, i // 100 % 10*0.1
        # if not (l and j and k):
        #     continue
        # logger.info(f'p1:{l}, p2:{j}, r_p:{k}')
        acc_list = []
        estp_acc_list = []
        model = ae(graph).to('cuda')
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / 1500)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=scheduler)
        # model.set_para(0.3, 0.4, 0.1)
        model.set_para(0.5, 0.3, 0.05)
        for epoch1 in range(3000):
            train_epoch(epoch1)
        logger.info("############################################")
        logger.info(f"# best_acc: {max(acc_list)} # best_early-stopping: {max(estp_acc_list)}")
        best_acc.append(max(acc_list))
        best_early_stopping.append(max(estp_acc_list))
    logger.info(f"# final_acc: {max(best_acc)} # final_early-stopping: {max(best_early_stopping)}")
    logger.info(
        f"# mean_acc: {sum(best_acc) / len(best_acc)} # mean_early-stopping: {sum(best_early_stopping) / len(best_early_stopping)}")
    logger.info("finish!")