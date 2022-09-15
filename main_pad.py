#Filename:	main.py
#institute: IIT Roorkee

import argparse
import os
import pandas as pd
import numpy as np
import copy 
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
    parser.add_argument("-cls", type = int, default = 6, help = "dataset classes")
    parser.add_argument("-gpu", type = bool, default = True, help = "Use gpu to accelerate")
    parser.add_argument("-batch_size", type = int, default = 16, help = "batch size for dataloader")
    parser.add_argument("-lr", type = float, default = 0.001, help = "initial learning rate")
    parser.add_argument("-epoch", type = int, default = 150, help = "training epoch")
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
    save_path = os.path.join(folder_path,'pad_model','epoch_{}.csv'.format(model_name))
    result.to_csv(save_path,index= False)

    return best_model, best_acc, best_top5, best_iters

if __name__ == "__main__":

    meta_data_columns = ["smoke_False", "smoke_True", "drink_False", "drink_True", "background_father_POMERANIA",
                         "background_father_GERMANY", "background_father_BRAZIL", "background_father_NETHERLANDS",
                         "background_father_ITALY", "background_father_POLAND",	"background_father_UNK",
                         "background_father_PORTUGAL", "background_father_BRASIL", "background_father_CZECH",
                         "background_father_AUSTRIA", "background_father_SPAIN", "background_father_ISRAEL",
                         "background_mother_POMERANIA", "background_mother_ITALY", "background_mother_GERMANY",
                         "background_mother_BRAZIL", "background_mother_UNK", "background_mother_POLAND",
                         "background_mother_NORWAY", "background_mother_PORTUGAL", "background_mother_NETHERLANDS",
                         "background_mother_FRANCE", "background_mother_SPAIN", "age", "pesticide_False",
                         "pesticide_True", "gender_FEMALE", "gender_MALE", "skin_cancer_history_True",
                         "skin_cancer_history_False", "cancer_history_True", "cancer_history_False",
                         "has_piped_water_True", "has_piped_water_False", "has_sewage_system_True",
                         "has_sewage_system_False", "fitspatrick_3.0", "fitspatrick_1.0", "fitspatrick_2.0",
                         "fitspatrick_4.0", "fitspatrick_5.0", "fitspatrick_6.0", "region_ARM", "region_NECK",
                         "region_FACE", "region_HAND", "region_FOREARM", "region_CHEST", "region_NOSE", "region_THIGH",
                         "region_SCALP", "region_EAR", "region_BACK", "region_FOOT", "region_ABDOMEN", "region_LIP",
                         "diameter_1", "diameter_2", "itch_False", "itch_True", "itch_UNK", "grew_False", "grew_True",
                         "grew_UNK", "hurt_False", "hurt_True", "hurt_UNK", "changed_False", "changed_True",
                         "changed_UNK", "bleed_False", "bleed_True", "bleed_UNK", "elevation_False", "elevation_True",
                         "elevation_UNK"]

    # Clear occupied memory
    gc.collect()
    torch.no_grad()
    torch.cuda.empty_cache() 
    torch.cuda.memory_summary(device=None, abbreviated=False)

    args = parse_param()
    print_param(args)
    print(args)
    _folder = 1
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 20 
    _folder_path = ""                 # Path to your working directory
    _model_name  = "resnet"           # name of the model
    _base_path = os.path.join(_folder_path,"PAD-UFES-20")
    
    # Path to Images folder     
    _imgs_folder_train = os.path.join(_base_path, "imgs") 

    # path to csv containing metadata information  
    # Metadata is prepared using methods specified in MetaBlock Paper 
    _csv_path_train = os.path.join(_base_path, "pad-ufes-20_parsed_folders.csv") 
    _csv_path_test = os.path.join(_base_path, "pad-ufes-20_parsed_test.csv")    

    csv_all_folders = pd.read_csv(_csv_path_train)    
    val_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder) ]
    train_csv_folder = csv_all_folders[ csv_all_folders['folder'] != _folder ]

    print('Size of training Data: ',len(train_csv_folder))
    print('Size of Validaton Data: ',len(val_csv_folder))

    # Train Transform
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(),]), p=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]) 

    print("- Loading training data...")
    train_imgs_id = train_csv_folder['img_id'].values
    train_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = list(train_csv_folder['diagnostic_number'].values)
    train_meta_data = train_csv_folder[meta_data_columns].values
    print("-- Using {} meta-data features".format(len(meta_data_columns)))

    print('Number of classes:  ',train_csv_folder['diagnostic_number'].nunique())
    train_dataset = meta_img_dataset(train_imgs_path, train_meta_data, train_labels, train_transform) 
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)
    
    # Validation Transform
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]) 
    val_imgs_id = val_csv_folder['img_id'].values
    val_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in val_imgs_id]
    val_labels = val_csv_folder['diagnostic_number'].values     
    val_meta_data = val_csv_folder[meta_data_columns].values

    # create evaluation data
    val_dataset = meta_img_dataset_test(val_imgs_path, val_meta_data, val_labels, val_transform) 
    val_loader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    csv_test = pd.read_csv(_csv_path_test) 
    test_meta_data = csv_test[meta_data_columns].values

    # Test Transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]) 
    test_imgs_id = csv_test['img_id'].values
    test_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in test_imgs_id]
    test_labels = csv_test['diagnostic_number'].values     
    print('Size of Test Data: ',len(csv_test))

    # create evaluation data
    test_dataset = meta_img_dataset_test(test_imgs_path, test_meta_data, test_labels, test_transform) 
    test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    # weighted cross-entropy loss function
    ser_lab_freq = train_csv_folder.groupby("diagnostic")["img_id"].count() 
    _labels_name = ser_lab_freq.index.values 
    _freq = ser_lab_freq.values
    _weights = (_freq.sum() / _freq).round(3) 
    print('ser_lab_freq:    ',ser_lab_freq)
    # specify the loss function
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())

    # specify the model
    if _model_name == 'resnet':
        from pad_model.resnet import *
    model = resnet_pad(224, 6, attention = True)
 
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
    best_model, best_acc, best_top5, best_iters = run(model, train_loader, val_loader, optimizer, loss_func, writer, scheduler_lr, epoch, _folder_path, _model_name)
    print("Best acc {} at iteration {}, Top 5 {}".format(best_acc, best_iters, best_top5))

    # ON TEST dATASET:
    test_loss, test_acc,test_bacc, top5, time_elapsed,prediction,real_labels = test(best_model, test_loader, loss_func, True, True)
    print("Test set: Loss {}, Accuracy {},BACC {} Top 5 {}, Time Elapsed {}".format(test_loss / len(test_loader.dataset), test_acc / len(test_loader.dataset),test_bacc, top5 / len(test_loader.dataset), time_elapsed))
    
    pred_csv = pd.DataFrame({'Labels':real_labels,'Prediction':prediction}) 
    record = {0:'ACK',1:'BCC',2:'MEL',3:'NEV',4:'SCC',5:'SEK'}
    pred_csv['Diagnostic'] = -1
    for i in range(len(pred_csv)):
        pred_csv.iloc[i,2] = record[pred_csv.iloc[i,0]]
    pred_csv_path = os.path.join(_folder_path,'pad_model','prediction_{}.csv'.format(_model_name))    
    
    # Prediction on Test Dataset
    pred_csv.to_csv(pred_csv_path,index = False)

    # save model
    print('Saving model {}'.format(_model_name)) 
    test_acc = round(test_acc / len(test_loader.dataset),3)
    model_path = os.path.join(_folder_path, 'pad_model', "{}_".format(_model_name) + str(test_acc) + "__" + str(round(test_bacc,3)) + ".pkl")
    torch.save(best_model, model_path)

