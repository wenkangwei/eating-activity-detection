
from tqdm import tqdm
from sklearn.metrics import classification_report,recall_score, precision_score
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import classification_report,recall_score, precision_score


def eval_model(model,dataloader,device="cpu"):
    correct = 0.
    total = 0.
    TP = 0
    FN = 0
    # without update
    with torch.no_grad():
        for samples, labels in dataloader:
            samples = samples.to(device)
            labels = labels.to(device)
            outputs = model(samples).to(device).squeeze()
            #print("Output: ", outputs)
            outputs = torch.round(torch.sigmoid(outputs))
            preds = outputs>=0.5
            preds = preds.to(dtype = torch.float)
            preds.requires_grad = False
#             _,preds = torch.max(outputs,1)
            for i in range(len(preds)):
                if preds[i] == 1 and labels[i] == 1:
                    TP += 1
                if preds[i] == 0 and labels[i] == 1:
                    FN += 1
            correct += torch.sum((preds == labels)).item()
            total += float(len(labels))
        acc =100 * correct/ total
        recall = TP/(TP+FN)
#         print("Evaluation Acc: %.4f %%,  Recall: %.4f "%(acc , recall))
    return acc, recall
            
            
            
            

def train_model(model,dataloader, optimizer, criterion,lrscheduler,device="cpu" , n_epochs=20,
                earlystopping=True, patience= 3, l1_enabled=True,checkpoint_name ="checkpoint.pt" ):
    loss_ls = [0.0]
    train_acc_ls = [0.0]
    valid_acc_ls = [0.0]
    valid_acc = 0.0
    loss =0.0
    train_acc = 0.0
    patience_count = 0
    best_val_score = 0.0
    prev_val_score = 0.0
    best_model = None
    
    train_dataloader, valid_dataloader = dataloader
    print("Training set batch amounts:", len(train_dataloader))
    print("Test set :", len(valid_dataloader))
    print("Start Training..")
    
    for e in range(n_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        running_correct = 0.0
        correct_cnt = 0.0
        total_cnt = 0.0
        TP = 0.
        FN = 0.
        model.train()
        for i, (samples, labels) in enumerate(train_dataloader):
            samples = samples.to(device)
            labels = labels.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            # reshape samples
            outputs = model(samples).squeeze()

            #print("Output: ", outputs, "label: ", labels)
            
            # Compute loss
            loss = criterion(outputs, labels)
            if l1_enabled:
                L1_loss = model.l1_loss(0.01).to(device)
                loss += L1_loss
            loss.backward()
            optimizer.step()
            
            # prediction
            #_,preds = torch.max(outputs,1)
            outputs = torch.round(torch.sigmoid(outputs))
            preds = outputs>=0.5
            preds = preds.to(dtype = torch.float)
            preds.requires_grad = False
            
            # Compute count of TP, FN
            for j in range(len(preds)):
                if preds[j] == 1. and labels[j] == 1.:
                    TP += 1
                if preds[j] == 0. and labels[j] == 1.:
                    FN += 1
            
            running_loss += loss.item()
            correct_cnt += torch.sum((preds == labels)).item()
            total_cnt += float(len(labels))
            batch_acc = 100. * (preds == labels).sum().item()/ float(len(labels))
            if i %50 ==0:
                #print("===> Batch: %d,  Batch_Loss: %.4f, Train Acc: %.4f %%,  Recall: %.f\n"%(i, loss,batch_acc, recall))
                pass

            
        
        # Compute accuracy and loss of one epoch
        epoch_loss = running_loss / len(train_dataloader)  
        epoch_acc = 100* correct_cnt/ total_cnt  # in percentage
        correct_cnt = 0.0
        total_cnt = 0.0
        train_recall = TP/(TP+FN)
        
        #Validation mode
        model.eval()
        valid_acc, valid_recall= eval_model(model,valid_dataloader,device=device)
        
        # record loss and accuracy
        valid_acc_ls.append(valid_acc)  
        train_acc_ls.append(epoch_acc)
        loss_ls.append(epoch_loss)
        
        if e %1==0:
            print("Epoch: %d,  Epoch_Loss: %.4f, Train Acc: %.4f %%, Train Recall: %.4f "%(e, epoch_loss,
                                                                                     epoch_acc,train_recall))
            print("Validation Acc:  %.4f %%,  Validation Recall: %.4f "%(valid_acc, valid_recall))
        
        # Reset train mode
        model.train()
        lrscheduler.step(valid_acc)
        
        
        # If earlystopping is enabled, then save model if performance is improved
        if earlystopping:
            if prev_val_score !=0. and valid_acc < prev_val_score :
                patience_count += 1
            else:
                patience_count = 0
                
            if patience_count >= patience:
                break 
                
            prev_val_score = valid_acc
            if valid_acc > best_val_score or best_val_score == 0.0:
                best_val_score = valid_acc
                torch.save(model,checkpoint_name)
                print("Checkpoint Saved")
            
                
        print("\n")
        
        
            
    # Load best model
    best_model = torch.load(checkpoint_name)
    print("Load Best Model.")
    print("Training completed")
        
    return model, best_model,best_val_score,loss_ls, train_acc_ls, valid_acc_ls
            

def plot_data(train_acc_ls,valid_acc_ls,loss_ls ):
    """
    Plot validation accuracy, training accuracy and loss
    """
    fig, ax = plt.subplots(1,2,figsize=(20,5))
    epochs = [i for i in range(len(train_acc_ls))]
    _ = sns.lineplot(x=epochs, y= train_acc_ls,ax=ax[0])
    _ = sns.lineplot(x=epochs, y= valid_acc_ls,ax=ax[0])
    ax[0].set_xlabel("Epoches")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(["Training Accuracy", "Validation Accuracy"])
    
    _ = sns.lineplot(x=epochs[1:], y= loss_ls[1:],ax=ax[1])
    ax[1].set_xlabel("Epoches")
    ax[1].set_ylabel("Training Loss")
    ax[1].set(yscale="log")
    plt.show()
    
    
            
