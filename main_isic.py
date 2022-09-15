#Filename:	main_isic.py
#Institute: IIT Roorkee

import argparse
import os
import copy 
import pandas as pd
import numpy as np
import torch.optim as optim
from torchvision import transforms
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter 
from train_test import *
import gc
from utils.custom_dataset import *

def parse_param():
    """
    parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cls", type = int, default = 8, help = "dataset classes")
    parser.add_argument("-gpu", type = bool, default = True, help = "Use gpu to accelerate")
    parser.add_argument("-batch_size", type = int, default = 4, help = "batch size for dataloader")
    parser.add_argument("-lr", type = float, default = 0.001, help = "initial learning rate")
    parser.add_argument("-epoch", type = int, default = 2, help = "training epoch")
    parser.add_argument("-optimizer", type = str, default = "sgd", help = "optimizer")
    args = parser.parse_args()

    return args

def print_param(args):
    """
    print the arguments
    """
    print("-" * 15 + "training configuration" + "-" * 15)
    print("class number:{}".format(args.cls))
    print("batch size:{}".format(args.batch_size))
    print("gpu used:{}".format(args.gpu))
    print("learning rate:{}".format(args.lr))
    print("training epoch:{}".format(args.epoch))
    print("optimizer used:{}".format(args.optimizer))
    print("-" * 53)

def run(model, train_loader, val_loader, optimizer, loss_func,  writer, train_scheduler, epoch, folder_path, model_name):
    best_acc = 0
    best_top5 = 0
    best_iters = 0
    best_model = model
    tn_loss, tn_acc, vl_loss, vl_acc = [],[],[],[]

    for i in range(epoch):
        gc.collect()
        torch.cuda.empty_cache() 
        torch.cuda.memory_summary(device=None, abbreviated=False)    
        
        print("Epoch {}".format(i))

        # performance on training set  
        model, train_loss, train_acc, time_elapsed = train(model, train_loader, loss_func, optimizer, True)
        print("Training set: Epoch {}, Loss {}, Accuracy {}, Time Elapsed {}".format(i, train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset), time_elapsed))
        writer.add_scalar("Train/loss", train_loss / len(train_loader.dataset), i)
        writer.add_scalar("Train/acc", train_acc / len(train_loader.dataset), i)

        tn_acc.append(train_acc / len(train_loader.dataset))
        tn_loss.append(train_loss / len(train_loader.dataset))

        # record the layers' gradient
        for name, param in model.named_parameters():
            if "weight" in name and not isinstance(param.grad, type(None)):
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram("{}/{}_grad".format(layer, attr), param.grad.clone().cpu().data.numpy(), i)

        # record the weights distribution
        for name, param in model.named_parameters():
            if "weight" in name:
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram("{}/{}".format(layer, attr), param.clone().cpu().data.numpy(), i)

        # performance on validation set
        val_loss, val_acc, top5, time_elapsed = test(model, val_loader, loss_func, True)
        print("Validation set: Epoch {}, Loss {}, Accuracy {},Top 5 {}, Time Elapsed {}".format(i, val_loss / len(val_loader.dataset), val_acc / len(val_loader.dataset), top5 / len(val_loader.dataset), time_elapsed))
        writer.add_scalar("Val/loss", val_loss / len(val_loader.dataset), i)
        writer.add_scalar("Val/acc", val_acc / len(val_loader.dataset), i)
        writer.add_scalar("Val/top5", top5 / len(val_loader.dataset), i)
    
        val_acc = float(val_acc) / len(val_loader.dataset)
        top5    = float(top5) / len(val_loader.dataset)

        vl_acc.append(val_acc)
        vl_loss.append(val_loss / len(val_loader.dataset))

        train_scheduler.step(val_loss / len(val_loader.dataset))
        
        print('Current Lr: ',optimizer.param_groups[0]['lr'])

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
            best_iters = i

        if top5 > best_top5:
            best_top5 = top5
    
    result = pd.DataFrame({'Train Accuracy':tn_acc,'Train Loss':tn_loss,'Validation Accuracy': vl_acc,'Validation Loss':vl_loss})
    save_path = os.path.join(folder_path,'isic_model','epoch_{}.csv'.format(model_name))
    result.to_csv(save_path,index= False)

    return best_model, best_acc, best_top5, best_iters



if __name__ == "__main__":

    meta_data_columns = ['age_approx', 'female', 'male', 'anterior torso', 'head/neck', "lateral torso",
                         'lower extremity', 'oral/genital', 'palms/soles', 'posterior torso',  'upper extremity']

    # Clear occupied memory
    gc.collect()
    torch.no_grad()
    torch.cuda.empty_cache() 
    torch.cuda.memory_summary(device=None, abbreviated=False)

    args = parse_param()
    print_param(args)
    print(args)

    _folder_val = 1
    _folder_test = 2 
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 20

    _folder_path = ""                      # Path to your working directory 
    _model_name  = "resnet"                # name of the model    
    _base_path = os.path.join(_folder_path,"ISIC")   
    # Images Path
    _imgs_folder_train = os.path.join(_base_path, "ISIC_2019_Training_Input")  
    # path to csv containing metadata information  
    # Metadata is prepared using methods specified in MetaBlock Paper 
    _csv_path_train = os.path.join(_base_path, "ISIC2019_parsed_train_15_folders.csv") 
 
    csv_all_folders = pd.read_csv(_csv_path_train)    
    val_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder_val) ] 
    test_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder_test) ] 
    train_csv_folder = csv_all_folders[(csv_all_folders['folder'] != _folder_val) & (csv_all_folders['folder'] != _folder_test)] 

    print('Size of training Data: ',len(train_csv_folder))
    print('Size of Validaton Data: ',len(val_csv_folder))
    print('Size of Test Data: ',len(test_csv_folder))  

    # Train Transform
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(),]), p=0.1), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]) 
 
    print("- Loading training data...")
    train_imgs_id = train_csv_folder['image'].values
    train_imgs_path = ["{}/{}.jpg".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder['diagnostic_number'].values
    train_meta_data = train_csv_folder[meta_data_columns].values
    print("-- Using {} meta-data features".format(len(meta_data_columns)))

    print('number of classes:  ',train_csv_folder['diagnostic_number'].nunique()) 
    train_dataset = meta_img_dataset(train_imgs_path, train_meta_data, train_labels, train_transform) 
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)
    
    # Validation Transform 
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

    # Loading validation data
    val_imgs_id = val_csv_folder['image'].values
    val_imgs_path = ["{}/{}.jpg".format(_imgs_folder_train, img_id) for img_id in val_imgs_id]
    val_labels = val_csv_folder['diagnostic_number'].values
    val_meta_data = val_csv_folder[meta_data_columns].values
 
    # create evaluation data
    val_dataset = meta_img_dataset_test(val_imgs_path, val_meta_data, val_labels, val_transform) 
    val_loader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    # Test Transform    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]) 

    # Loading test data    
    test_imgs_id = test_csv_folder['image'].values
    test_imgs_path = ["{}/{}.jpg".format(_imgs_folder_train, img_id) for img_id in test_imgs_id]
    test_labels = test_csv_folder['diagnostic_number'].values
    test_meta_data = test_csv_folder[meta_data_columns].values

    # create evaluation data
    test_dataset = meta_img_dataset_test(test_imgs_path, test_meta_data, test_labels, test_transform) 
    test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    # Cost Function
    ser_lab_freq = train_csv_folder.groupby("diagnostic")["image"].count() 
    _labels_name = ser_lab_freq.index.values 
    _freq = ser_lab_freq.values
    _weights = (_freq.sum() / _freq).round(3) 
    print('ser_lab_freq:    ',ser_lab_freq)
    # specify the loss function 
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())

    # specify the model
    if _model_name == 'resnet': 
        from isic_model.resnet import *
    model = resnet_isic(224, 8, attention = True)

    print('Number of Cuda Device Available:  ',torch.cuda.device_count())

    # specify gpu used
    if args.gpu == True:
        model = model.cuda()
        loss_func = loss_func.cuda()
    
    # specify optimizer
    if args.optimizer == 'sgd':
        print('SGD Optimizer')
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 0.001)
    elif args.optimizer == 'adam':
        print('Adam Optimizer')
        optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)  
    else:
        optimizer = optim.momentum(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 0.002)
    
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience)

    # specify the epoches
    epoch = args.epoch

    Time = "{}".format(datetime.now().isoformat(timespec='seconds')).replace(':', '-')
    writer = SummaryWriter(log_dir = os.path.join("./log/", Time))

    best_model, best_acc,best_top5, best_iters = run(model, train_loader, val_loader, optimizer, loss_func, writer, scheduler_lr, epoch,_folder_path, _model_name)
    print("Best acc {} at iteration {}, Top 5 {}".format(best_acc, best_iters, best_top5))

    # ON TEST dATASET:
    test_loss, test_acc,test_bacc, top5, time_elapsed,prediction,real_labels = test(best_model, test_loader, loss_func, True,True)
    print("Test set: Loss {}, Accuracy {},BACC {} Top 5 {}, Time Elapsed {}".format(test_loss / len(test_loader.dataset), test_acc / len(test_loader.dataset),test_bacc, top5 / len(test_loader.dataset), time_elapsed))
    
    pred_csv = pd.DataFrame({'Labels':real_labels,'Prediction':prediction}) 
    record = {0:'AK',1:'BCC',2:'BKL',3:'DF',4:'MEL',5:'NV', 6 : 'SCC', 7 : 'VASC'}
    pred_csv['Diagnostic'] = -1 
    for i in range(len(pred_csv)):
        pred_csv.iloc[i,2] = record[pred_csv.iloc[i,0]]

    pred_csv_path = os.path.join(_folder_path,'isic_model','prediction_{}.csv'.format(_model_name))   

    # Prediction on test dataset 
    pred_csv.to_csv(pred_csv_path,index = False)

    # save model
    print('Saving Model {}'.format(_model_name)) 
    test_acc = round(test_acc / len(test_loader.dataset),3)
    model_path = os.path.join(_folder_path,'isic_model', "{}_".format(_model_name) + str(test_acc) + "__" + str(round(test_bacc,3)) + ".pkl")
    torch.save(best_model, model_path)

