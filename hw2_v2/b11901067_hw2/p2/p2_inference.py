# ============================================================================
# File: p2_inference.py
# Date: 2025-03-11
# Author: TA
# Description: Load pre-trained model and perform inference on test set.
# ============================================================================
import os
import sys
import time
import argparse
import numpy as np
import torch

from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import write_csv

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_datadir',
                        help='test dataset directory',
                        type=str,
                        default='../hw2_data/p2_data/val/')
    parser.add_argument('--model_type',
                        help='mynet or resnet18',
                        type=str,
                        default='resnet18')
    parser.add_argument('--output_path',
                        help='output csv file path',
                        type=str,
                        default='./output/pred.csv')
    args = parser.parse_args()

    model_type = args.model_type
    test_datadir = args.test_datadir
    output_path = args.output_path

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ##### MODEL #####
    ##### TODO: check model.py #####
    ##### NOTE: Put your best trained models to checkpoint/ #####
    if model_type == 'mynet':
        model = MyNet()
        model.load_state_dict(torch.load('./checkpoint/mynet_best.pth', 
                                         map_location=torch.device('cpu')))


    elif model_type == 'resnet18':
        model = ResNet18()
        model.load_state_dict(torch.load('./checkpoint/resnet18_best.pth', 
                                         map_location=torch.device('cpu')))
    else:
        raise NameError('Unknown model type')
    model.to(device)
    # print(model)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")

    ##### DATALOADER #####
    ##### TODO: check dataset.py #####
    test_loader = get_dataloader(test_datadir, batch_size=1, split='test')

    ##### INFERENCE #####
    predictions = []
    model.eval()  # 設定模型為評估模式 (evaluation mode)
    with torch.no_grad():  # 在測試時不計算梯度，節省記憶體
        test_start_time = time.time()
        
        for batch in test_loader:  # 迭代測試資料
            images = batch['images'].to(device)  # 從 dict 提取 'images'

            
            outputs = model(images)  # 前向傳播
            _, predicted = torch.max(outputs, 1)  # 取得最大值的索引 (即預測類別)
            
            predictions.extend(predicted.cpu().numpy())  # 轉成 NumPy 並存入 predictions


    test_time = time.time() - test_start_time
    print()
    print(f'Finish testing {test_time:.2f} sec(s), dumps result to {output_path}')

    ##### WRITE RESULT #####
    write_csv(output_path, predictions, test_loader)
    
if __name__ == '__main__':
    main()
