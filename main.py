import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from faceformer import Faceformer

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    # 1️⃣ 组合模型权重保存的目录，如 vocaset/save
    save_path = os.path.join(args.dataset, args.save_path)

    # 2️⃣ 如果目录已存在就整块删除，保证每次训练都是全新的输出
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    # 3️⃣ 再创建一个干净的保存目录
    os.makedirs(save_path)

    # 4️⃣ 训练用到的“说话人列表”拆成 Python list，后面做条件生成用
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    iteration = 0                     # 5️⃣ 统计全局迭代步（for 进度条显示）

    # ---------------------- 主循环：跑 epoch ----------------------
    for e in range(epoch + 1):        # 注意是 epoch+1，所以包含第 0~epoch 总共 epoch+1 轮
        loss_log = []                 # 6️⃣ 记录本轮所有 batch 的 loss，用来取均值显示

        # ---------- 训练阶段 ----------
        model.train()                 # 7️⃣ 切换到训练模式（启用 dropout / BN 写入）
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # 8️⃣ tqdm 进度条
        optimizer.zero_grad()         # 9️⃣ 先清一次梯度，配合梯度累积

        # 10️⃣ 遍历一个 epoch 内的所有 batch
        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1

            # 11️⃣ 把数据搬到 GPU
            audio     = audio.to(device="cuda")
            vertice   = vertice.to(device="cuda")
            template  = template.to(device="cuda")
            one_hot   = one_hot.to(device="cuda")

            # 12️⃣ 前向 + 计算损失
            #     model(...) 本质执行 Faceformer.forward(...)
            loss = model(audio, template, vertice, one_hot,
                         criterion, teacher_forcing=False)

            # 13️⃣ 反向传播，梯度累积
            loss.backward()

            # 14️⃣ 记录当前 batch 的 loss 数值（CPU 标量）
            loss_log.append(loss.item())

            # 15️⃣ 每累积 gradient_accumulation_steps 个 batch 就做一次参数更新
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()      # 更新权重
                optimizer.zero_grad() # 清梯度

            # 16️⃣ 更新 tqdm 状态栏，显示平均 loss
            pbar.set_description(
                f"(Epoch {e+1}, iteration {iteration}) "
                f"TRAIN LOSS:{np.mean(loss_log):.7f}"
            )

        # ---------- 验证阶段 ----------
        valid_loss_log = []           # 17️⃣ 记录验证集 loss
        model.eval()                  # 18️⃣ 切到评估模式（停用 dropout / BN 写入）

        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            # 19️⃣ GPU 化
            audio, vertice, template, one_hot_all = (
                audio.to("cuda"), vertice.to("cuda"),
                template.to("cuda"), one_hot_all.to("cuda")
            )

            # 20️⃣ 取出当前文件属于哪个说话人
            train_subject = "_".join(file_name[0].split("_")[:-1])

            if train_subject in train_subjects_list:
                # 21️⃣ 如果这个说话人在训练集里，就用它自己的 one‑hot 条件
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]
                loss = model(audio, template, vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                # 22️⃣ 否则：对每个训练说话人都做一次条件生成并算 loss
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:, iter, :]
                    loss = model(audio, template, vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())

        # 23️⃣ 计算本 epoch 验证集平均 loss
        current_loss = np.mean(valid_loss_log)

        # ---------- 按间隔保存模型 ----------
        # 24️⃣ 每 25 个 epoch 或最后一轮保存一次
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(save_path, f"{e}_model.pth")
            )

        # 25️⃣ 打印验证集 loss
        print(f"epcoh: {e+1}, current loss:{current_loss:.7f}")

    # 26️⃣ 训练完返回最终模型（参数已更新）
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = os.path.join(args.dataset,args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()
   
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    #build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["valid"],model, optimizer, criterion, epoch=args.max_epoch)
    
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()