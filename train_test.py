#Filename:	train_test.py
#Instiute:  IIT Roorkee

from torch.autograd import Variable
import torch
import time
import gc

def train(model, train_loader, loss_func, optimizer, is_gpu=True):
    # Clear occupied memory
    gc.collect()
    torch.cuda.empty_cache() 
    torch.cuda.memory_summary(device=None, abbreviated=False)    
    model.train()
    start = time.time()
    epoch_loss = 0
    epoch_acc = 0
    for step, (imgs_batch, metadata_batch, batch_y) in enumerate(train_loader):
        if is_gpu:
            img_inputs, metadata_inputs, labels = Variable(imgs_batch.float().cuda()), Variable(metadata_batch.float().cuda()), Variable(batch_y.long().cuda())
        else:
            img_inputs,  metadata_inputs, labels = Variable(imgs_batch.float()), Variable(metadata_batch.float()), Variable(batch_y.long())
        optimizer.zero_grad() 
        outputs = model(img_inputs, metadata_inputs)
        _, preds = torch.max(outputs[0].data, 1)
        loss = loss_func(outputs[0], labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        epoch_acc += torch.sum(preds == labels.data).item()
        

    end = time.time()
    time_elapsed = end - start

    return model, epoch_loss, epoch_acc, time_elapsed


def test(model, test_loader, loss_func, is_gpu = True,return_pred = False):
    start = time.time()
    epoch_loss = 0
    epoch_acc = 0
    top5 = 0
    mask = max((1, 5))
    model.eval()
    prediction,real_labels = [],[]
    for step, (imgs_batch, metadata_batch, batch_y) in enumerate(test_loader):
        # wrap them in Variable
        if is_gpu:
            img_inputs, metadata_inputs, labels = Variable(imgs_batch.float().cuda()), Variable(metadata_batch.float().cuda()), Variable(batch_y.long().cuda())
        else:
            img_inputs,  metadata_inputs, labels = Variable(imgs_batch.float()), Variable(metadata_batch.float()), Variable(batch_y.long())
        outputs = model(img_inputs, metadata_inputs)
        _, preds = torch.max(outputs[0].data, 1)
         
        if return_pred:
            prediction = prediction +  preds.tolist()
            real_labels = real_labels + labels.tolist()

        loss = loss_func(outputs[0], labels)

        epoch_loss += loss.detach().item()
        epoch_acc += torch.sum(preds == labels.data).item()
        _, top5_preds = outputs[0].topk(mask, 1, True, True)
        # compute the top-5 acc
        for i in range(len(imgs_batch)):
            if labels[i] in top5_preds[i]:
                top5 += 1
        
    end = time.time()
    time_elapsed = end - start
    if return_pred:
        import sklearn.metrics as skmet
        test_bacc = skmet.balanced_accuracy_score(real_labels,prediction)  
        return epoch_loss, epoch_acc,test_bacc, top5, time_elapsed,prediction,real_labels
    else:
        return epoch_loss, epoch_acc, top5, time_elapsed

