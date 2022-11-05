# cedro blog
# PyTorch まずMLPを使ってみる
# http://cedro3.com/ai/pytorch-mlp/
#

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms

import argparse

from common.tools import mkdir
from common.randomControl import torch_fix_seed

from KS_mlp import MLPNet
from KS_nsl_kdd import NSL_KDD
from KS_MyDataset import MyDataset


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0 
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        y = y.squeeze()  ####
        loss = loss_fn(pred, y)

        train_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            y = y.squeeze()  ####
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default = -1)  
    parser.add_argument("-dd", "--dirDataset", type=str, default="~/Research2/dataset")  
    parser.add_argument("-n", "--numOfLabels", type=int, choices=[2, 5], default=2)  
    parser.add_argument("-m", "--numOfNeurons", type=int, default = 32)  
    parser.add_argument("-bs", "--batchSize", type=int, default = 128)  
    parser.add_argument("-ep", "--epochs", type=int, default = 50)  
    parser.add_argument("-dp", "--dir_pic", type=str, default="pic")  # グラフ結果出力先ディレクトリ
    parser.add_argument("-pj", "--pjName", type=str, default="pjName")  # Project name
    # parser.add_argument('--en', action='store_true') # if --en exist; True, otherwise False
    # parser.add_argument('--en', action='store_false') # if --en exist ; False, otherwise True
    parser.add_argument("-x2","--inclFlgFG2", action="store_false", help="exclude group2 features") # if this option is set, inclFlgFG becomes False
    parser.add_argument("-x3","--inclFlgFG3", action="store_false", help="exclude group3 features") # if this option is set, inclFlgFG becomes False
    parser.add_argument("-x4","--inclFlgFG4", action="store_false", help="exclude group4 features") # if this option is set, inclFlgFG becomes False
    parser.add_argument("-x5","--inclFlgFG5", action="store_false", help="exclude group5 features") # if this option is set, inclFlgFG becomes False
    parser.add_argument("-x6","--inclFlgFG6", action="store_false", help="exclude group6 features") # if this option is set, inclFlgFG becomes False

    args = parser.parse_args()
    seed = args.seed
    if (args.seed >= 0):
        print(f"random seed is {seed}.")
        torch_fix_seed(seed)
    else:
        print("random seed is NOT fixed.")
    epochs = args.epochs
    dirDataset = args.dirDataset+'/NSL-KDD'
    dir_pic = args.dir_pic
    mkdir(dir_pic)
    pjName = args.pjName
    numOfLabels = args.numOfLabels
    if (numOfLabels != 2) and (numOfLabels != 5):
        raise Exception("unsupported number of labels.")
    numOfNeurons = args.numOfNeurons
    print(f"num of Neurons is {numOfNeurons}.")
    batchSize = args.batchSize
    print(f"batch size is {batchSize}.")
    inclFlgFG2 = args.inclFlgFG2
    inclFlgFG3 = args.inclFlgFG3
    inclFlgFG4 = args.inclFlgFG4
    inclFlgFG5 = args.inclFlgFG5
    inclFlgFG6 = args.inclFlgFG6
    #
    file_path_20_percent = dirDataset+'/KDDTrain+_20Percent.txt'
    file_path_full_training_set = dirDataset+'/KDDTrain+.txt'
    file_path_test = dirDataset+'/KDDTest+.txt' 
    #
    dfGen = NSL_KDD(fnameTrain = file_path_full_training_set,
                    fnameTest = file_path_test,
                    numOfLabels = numOfLabels,
                    inclFlgFG2 = inclFlgFG2,
                    inclFlgFG3 = inclFlgFG3,
                    inclFlgFG4 = inclFlgFG4,
                    inclFlgFG5 = inclFlgFG5,
                    inclFlgFG6 = inclFlgFG6)
    trainDf, testDf = dfGen.getDf()
    featureNames = dfGen.getFeatures()
    labelNames = dfGen.getLabels()
    # #
    # dfGen.show()
    # dfGen.showDfInputFile()
    
    # train_Dataset = MyDataset(df=trainDf, features=featureNames, labels=labelNames, transform=None)
    train_dataset = MyDataset(df=trainDf, features=featureNames, labels=labelNames, transform=None)
    test_dataset = MyDataset(df=testDf, features=featureNames, labels=labelNames, transform=None)
    # sizeTrainDataset = len(train_Dataset)
    sizeTrainDataset = len(train_dataset)
    sizeTrain = int(0.3*sizeTrainDataset)
    sizeTest = sizeTrainDataset-sizeTrain
    print(f"train:test={sizeTrain}:{sizeTest}.")
    # train_dataset, valid_dataset = torch.utils.data.random_split(  # データセットの分割
    #     train_Dataset,   # 分割するデータセット
    #     [sizeTrain, sizeTest])  # 分割数 

    # set data loader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,  # データセットの指定
        batch_size=batchSize,  # ミニバッチの指定
        shuffle=True,  # シャッフルするかどうかの指定
        num_workers=2)  # コアの数

    # valid_dataloader = torch.utils.data.DataLoader(
    #     dataset=valid_dataset,
    #     batch_size=batchSize, 
    #     shuffle=False,
    #     num_workers=2)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batchSize, 
        shuffle=False,
        num_workers=2)

    if (epochs < 0):
        print("all dataset is used.")
        # epochs = len(train_dataset)/batchSize
        epochs = -(-len(train_dataset) // batchSize)
        print(f"size of train dataset: {len(train_dataset)}, batch size: {batchSize}")
        print(f"epoch is {epochs}.")

    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPNet(numOfFeatures = len(featureNames),
                   numOfLabels = numOfLabels,
                   numOfNeurons = numOfNeurons).to(device)


    #######################################################
    #######################################################
    #######################################################
    learning_rate = 1e-3
    batch_size = 64

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)

    for t in range(epochs):
        print("-------------------------------")
        print(f"Epoch {t+1}/{epochs}")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    #######################################################
    #######################################################
    #######################################################
    


    # # optimizing
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(),
    #                       lr = 1.0e-4,
    #                       momentum = 0.9,
    #                       weight_decay = 5.0e-4)

    # ###  training
    # print ('training start ...')

    # # initialize list for plot graph after training
    # train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

    # for epoch in range(epochs):
    #     # initialize each epoch
    #     train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    
    #     # ======== train mode ======
    #     model.train()
    #     for i, (images, labels) in enumerate(train_dataloader):  # ミニバッチ回数実行
    #         ##viewで28×28×１画像を１次元に変換し、deviceへ転送
    #         # images, labels = images.view(-1, 28*28*1).to(device), labels.to(device)
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()  # 勾配リセット
    #         outputs = model(images)  # 順伝播の計算
    #         labels = labels.squeeze()
    #         loss = criterion(outputs, labels)  # lossの計算
    #         train_loss += loss.item()  # train_loss に結果を蓄積
    #         acc = (outputs.max(1)[1] == labels).sum()  #  予測とラベルが合っている数の合計
    #         train_acc += acc.item()  # train_acc に結果を蓄積
    #         loss.backward()  # 逆伝播の計算        
    #         optimizer.step()  # 重みの更新
    #         avg_train_loss = train_loss / len(train_dataloader.dataset)  # lossの平均を計算
    #         avg_train_acc = train_acc / len(train_dataloader.dataset)  # accの平均を計算
    
    #     # ======== valid mode ======
    #     model.eval()
    #     with torch.no_grad():  # 必要のない計算を停止
    #         for images, labels in valid_dataloader:        
    #             # images, labels = images.view(-1, 28*28*1).to(device), labels.to(device)
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             labels = labels.squeeze()
    #             loss = criterion(outputs, labels)
    #             val_loss += loss.item()
    #             acc = (outputs.max(1)[1] == labels).sum()
    #             val_acc += acc.item()
    #     avg_val_loss = val_loss / len(valid_dataloader.dataset)
    #     avg_val_acc = val_acc / len(valid_dataloader.dataset)
    
    #     # print log
    #     print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
    #            .format(epoch+1, epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

    #     # append list for plot graph after training
    #     train_loss_list.append(avg_train_loss)
    #     train_acc_list.append(avg_train_acc)
    #     val_loss_list.append(avg_val_loss)
    #     val_acc_list.append(avg_val_acc)
 
    # # ======== fainal test ======
    # model.eval()
    # with torch.no_grad():
    #     total = 0
    #     test_acc = 0
    #     for images, labels in test_dataloader:        
    #         # images, labels = images.view(-1, 28 * 28 * 1 ).to(device), labels.to(device)
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         labels = labels.squeeze()
    #         test_acc += (outputs.max(1)[1] == labels).sum().item()
    #         total += labels.size(0)
    #     print('test_accuracy: {} %'.format(100 * test_acc / total)) 

    # # save weights
    # torch.save(model.state_dict(), 'mnist_model.ckpt')

    # # plot graph
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(range(epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    # plt.plot(range(epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('Training and validation loss')
    # # plt.grid()
    # #
    # plt.savefig(f'{dir_pic}/TrainValidationLoss_{pjName}.pdf')
    # plt.show()
    
