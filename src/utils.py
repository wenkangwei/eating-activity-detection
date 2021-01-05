import sys
import os
import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report,recall_score, precision_score
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import classification_report,recall_score, precision_score


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


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
            outputs = model(samples).squeeze()
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
                earlystopping=True, patience= 5, l1_enabled=True,checkpoint_name ="checkpoint.pt" ):
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
            print("Epoch: %d,  Epoch_Loss: %.4f, Train Acc: %.4f %%, Train Recall: %.4f, Validation Acc:  %.4f %%,  Validation Recall: %.4f  "%(e, epoch_loss,
                                                                                     epoch_acc,train_recall,valid_acc, valid_recall))
        
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
    
def split_train_test_indices(X, y, test_size, random_seed = None):
    """
    This function is to split the training set indices into validation set indices and training set indices
    
    X: indices of dataset/ subset of dataset
    y: labels of dataset / subset of dataset
    
    """
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split

    train_indices = []
    test_indices = []
    y_train = []
    y_test = []
    
    if test_size ==0:
        train_indices = X
        y_train = y
    elif test_size == 1:
        test_indices = X
        y_test = y
    elif test_size >0 and test_size <1:
        train_indices, test_indices, y_train, y_test = train_test_split(X,y,
                                                            stratify=y, 
                                                            test_size=test_size,random_state = random_seed)
        
    else:
        print("Invalid split ratio: %.3f"%(test_size))
    if len(train_indices)>0:
        print("Train set size: %d, with %d positive samples and %d negative samples"%(len(y_train),sum(y_train==1),
                                                                          sum(y_train==0)))
    if len(test_indices)>0:
        print("Test set size: %d, with %d positive samples and %d negative samples"%(len(y_test),
                                                                          sum(y_test==1),
                                                                           sum(y_test==0)))
    
    return train_indices, test_indices
            

    
def print_settings(winmin,stridesec, EPOCHS):
    """
    This is just a function to print information of training settings
    """
    outfile = sys.stdout

    winlength = int(winmin * 60 * 15)
    step = int(stridesec * 15)
    start_time = datetime.datetime.now()
    arr = ["echo -n 'PBS: node is '; cat $PBS_NODEFILE",\
          "echo PBS: job identifier is $PBS_JOBID",\
          "echo PBS: job name is $PBS_JOBNAME"]

    [os.system(cmd) for cmd in arr]
    print("*****************************************************************\n", file=outfile, flush=True)
    print("Execution Started at " + start_time.strftime("%m/%d/%Y, %H:%M:%S"), file=outfile, flush=True)
    print("WindowLength: {:.2f} min ({:d} datum)\tSlide: {:d} ({:d} datum)\tEpochs:{:d}\n".format(winmin, winlength, stridesec, step, EPOCHS), file=outfile, flush=True)
    
    

def cross_validation(dataset, data_indices, model,n_epochs=30,k=5, device="cpu", random_state = 1000, checkpoint_path = "./"  ):
    from sklearn.model_selection import StratifiedKFold
    
    best_val_score = 0
    overall_best_model = None
    best_fold = None
    all_loss_ls = []
    all_train_acc_ls = []
    all_valid_acc_ls = []
    data_indices = np.array(data_indices)
    
    skf = StratifiedKFold(n_splits=k)
    
    labels = dataset.labels[data_indices]
    np.random.seed(random_state)
    seeds = np.random.randint(low=0, high=1000,size=k)
    
    
    for fold_ind, (train_fold, valid_fold) in enumerate(skf.split(data_indices, labels)):
        torch.manual_seed(seeds[fold_ind])
        
        print("===========================> Running Fold: %d"%(fold_ind))
        print()
        train_indices = data_indices[train_fold]
        valid_indices = data_indices[valid_fold]
        # Train set    
        train_set_fold = torch.utils.data.Subset(dataset, train_indices)
        train_loader_fold = torch.utils.data.DataLoader(train_set_fold,batch_size=32, shuffle=True)

        # validation set
        valid_set_fold = torch.utils.data.Subset(dataset, valid_indices)
        valid_loader_fold = torch.utils.data.DataLoader(valid_set_fold,batch_size=32, shuffle=True)
          
        # Re-initialize models
        cv_model = model
        # Since I use a dynamic created layer in network, need to input a sample to initialize the model first
        cv_model.apply(weights_init)
        cv_model.to(device)
        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(cv_model.parameters(),lr=0.01,  weight_decay=0.1)
        lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience= 2, factor = 0.1,threshold=0.01)


        dataloader = (train_loader_fold,valid_loader_fold )
        cv_model, best_model,val_score,loss_ls, train_acc_ls, valid_acc_ls = train_model(cv_model,dataloader, optimizer, criterion, 
                                                                                      lrscheduler, device= device,
                                                                            n_epochs=n_epochs, patience = 5, l1_enabled=False,
                                                                            checkpoint_name =checkpoint_path+"cross_valid_checkpoint_"+str(fold_ind)+".pt")
        best_model.eval()
        valid_acc, recall = eval_model(best_model, valid_loader_fold,device)
        
        all_valid_acc_ls.append(valid_acc)
        
        print("Fold %d Completed"%(fold_ind))
    print("Cross Validation Completed，score is %.4f %%"%( np.mean(all_valid_acc_ls)))
    
    return all_valid_acc_ls
        

    
        
    
    
    

    
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.)
#         nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
#         nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.)
        nn.init.constant_(m.bias.data, 0)

        
        
def test_model(model, winmin=3, stridesec = 15,names= ["wenkanw"],random_seed=1000, split_day=False):
    """
    A function to test tensorflow model
    """
    perf = {"name":[],"model":[],"win(sec)":[], "acc":[],"recall":[], "auc":[]}
    for name in names:
        person = name
        if split_day:
            meal_data_test = Person_MealsDataset(person_name= person, file_name = "test_files", winmin = winmin,stridesec = stridesec)

            # balance test set
            testset_labels = meal_data_test.labels
            test_indices = balance_data_indices(testset_labels,data_indices=[i for i in range(len(meal_data_test))] ,mode="under", shuffle=True,random_state = random_seed,replace= False)
            # get numpy dataset
            test_Data, test_Labels = meal_data_test.get_subset(test_indices)
        else:            
            meal_data = Person_MealsDataset(person_name= person, file_name = "all_files_list", winmin = winmin,stridesec = stridesec)
            samples,labels =  meal_data.data_indices, meal_data.labels
            # split train set and test set
            train_indices, test_indices = split_train_test_indices(X= [i for i in range(len(labels))],
                                                                            y = labels, test_size = 0.2,
                                                                           random_seed = random_seed)
            testset_labels = labels[test_indices]
            test_indices = balance_data_indices(testset_labels,data_indices= test_indices,mode="under", shuffle=True,random_state = random_seed,replace= False)
            test_Data, test_Labels = meal_data.get_subset(test_indices)
            
        from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
        predictions = model.predict(x=test_Data)
        threshold = 0.5
        acc =  accuracy_score(predictions>=threshold,test_Labels)
        recall = recall_score(predictions>=threshold,test_Labels)
        auc = roc_auc_score(predictions>=threshold,test_Labels)
        print("Test Accuracy:", acc)
        print("Recall Accuracy:", recall)
        print("AUC Score:", auc)
        perf["name"].append(name)
        perf["model"].append("ActiModel")
        perf["win(sec)"].append(winmin*60)
        perf["acc"].append(acc)
        perf["recall"].append(recall)
        perf["auc"].append(auc)

    perf_df = pd.DataFrame(perf)
    return perf_df

def train_models_v2(model, win_ls = [],EPOCHS = 10,stridesec = 1,name = "wenkanw",model_name="v2" ,random_seed= 1000, split_day=False):
    """
    A function to train tensorflow models
    """
    from numpy.random import seed
    seed(random_seed)
    random.seed(random_seed)
#     tf.set_random_seed(random_seed)
    from datetime  import datetime
    batch_size = 128
    outfile = sys.stdout
    perf = {"model":[],"win(sec)":[], "acc":[],"recall":[], "auc":[]}
    model_ls = []
    hist_ls = []
    for winsize in win_ls:
        tf.random.set_seed(random_seed)
        seed(random_seed)
        
        winmin = winsize
        winlength = int(winmin * 60 * 15)
        step = int(stridesec * 15)
        start_time = datetime.now()
        arr = ["echo -n 'PBS: node is '; cat $PBS_NODEFILE",\
              "echo PBS: job identifier is $PBS_JOBID",\
              "echo PBS: job name is $PBS_JOBNAME"]
        [os.system(cmd) for cmd in arr]
        print("*****************************************************************\n", file=outfile, flush=True)
        print("Execution Started at " + start_time.strftime("%m/%d/%Y, %H:%M:%S"), file=outfile, flush=True)
        print("WindowLength: {:.2f} min ({:d} datum)\tSlide: {:d} ({:d} datum)\tEpochs:{:d}\n".format(winmin, winlength, stridesec, step, EPOCHS), file=outfile, flush=True)


        pathtemp = "../models/" + name+"_models" +"/"+model_name+"_M_F_"
        modelpath = pathtemp + "{:f}Min.h5".format(winmin)
        jsonpath = pathtemp + "{:f}Min.json".format(winmin)
        print("Model to Save: ",modelpath)
        print()
        # Load the dataset
        
        person = name
        if split_day:
            create_train_test_file_list(file_name= "all_files_list.txt",person_name =name,
                         out_path = "../data-file-indices/",root_path= "../",
                         test_ratio = 0.2, print_flag = True, shuffle=True, random_state=random_seed)

            meal_data_train = Person_MealsDataset(person_name= person, file_name = "train_files", winmin = winmin,stridesec = stridesec)
            meal_data_test = Person_MealsDataset(person_name= person, file_name = "test_files", winmin = winmin,stridesec = stridesec)

            train_indices, valid_indices = split_train_test_indices(X= [i for i in range(len(meal_data_train.labels))],
                                                                    y = meal_data_train.labels, test_size = 0.2,
                                                                   random_seed = random_seed)
            #balanced train set
            trainset_labels = meal_data_train.labels[train_indices]
            train_indices = balance_data_indices(trainset_labels,data_indices= train_indices,mode="under", shuffle=True,random_state = random_seed,replace= False)

            # balance test set
            testset_labels = meal_data_test.labels
            test_indices = balance_data_indices(testset_labels,data_indices=[i for i in range(len(meal_data_test))] ,mode="under", shuffle=True,random_state = random_seed,replace= False)
            # get numpy dataset
            balancedData, balancedLabels = meal_data_train.get_subset(train_indices)
            valid_balancedData, valid_balancedLabels = meal_data_train.get_subset(valid_indices)
            test_Data, test_Labels = meal_data_test.get_subset(test_indices)

        else:
        
            meal_data = Person_MealsDataset(person_name= person, file_name = "all_files_list", winmin = winmin,stridesec = stridesec)
            samples,labels =  meal_data.data_indices, meal_data.labels
            # split train set and test set
            train_indices, test_indices = split_train_test_indices(X= [i for i in range(len(labels))],
                                                                    y = labels, test_size = 0.2,
                                                                   random_seed = random_seed)
            # balance train set
            trainset_labels = labels[train_indices]
            train_indices_balanced = balance_data_indices(trainset_labels,data_indices= train_indices,mode="under", shuffle=True,random_state = random_seed,replace= False)
            #balance test set
            testset_labels = labels[test_indices]
            test_indices = balance_data_indices(testset_labels,data_indices= test_indices,mode="under", shuffle=True,random_state = random_seed,replace= False)

            train_set_balanced = torch.utils.data.Subset(meal_data, train_indices_balanced)
            test_set = torch.utils.data.Subset(meal_data, test_indices)

            train_loader = torch.utils.data.DataLoader(train_set_balanced,batch_size=batch_size, shuffle=True,num_workers=2)
            test_loader = torch.utils.data.DataLoader(test_set ,batch_size=batch_size, shuffle=True,num_workers=2)

            print("Data Loader Created")            
            
            # split validation set
            balanced_trainset_labels = labels[train_indices_balanced]
            train_indices, valid_indices = split_train_test_indices(X= train_indices_balanced,
                                                                    y = balanced_trainset_labels, test_size = 0.2,
                                                                   random_seed = random_seed)
            valid_set_balanced = torch.utils.data.Subset(meal_data, valid_indices)
            valid_loader = torch.utils.data.DataLoader(valid_set_balanced,batch_size=batch_size, shuffle=True,num_workers=2)

            balancedData, balancedLabels = meal_data.get_subset(train_indices)
            valid_balancedData, valid_balancedLabels = meal_data.get_subset(valid_indices)
            test_Data, test_Labels = meal_data.get_subset(test_indices)
        

        #training settings
        mcp_save = tf.keras.callbacks.ModelCheckpoint(modelpath, save_best_only=True, monitor='accuracy')
        

        scheduler = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=3, verbose=0,
                                             mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.)
        #train model
        H = model.fit(x=balancedData, y = balancedLabels,
                       validation_data=(valid_balancedData, valid_balancedLabels),
                    epochs = EPOCHS, batch_size=batch_size, verbose=1,
                    callbacks=[mcp_save,scheduler]) # removed addons.LossHistory(jsonpath) for compatibility with TensorFlow 2.2.0, needs to be re-added at some point

        print("Max value: ", max(H.history['accuracy']), " at epoch", np.argmax(H.history['accuracy']) + 1)

        from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
        predictions = model.predict(x=test_Data)
        threshold = 0.5
        acc =  accuracy_score(predictions>=threshold,test_Labels)
        recall = recall_score(predictions>=threshold,test_Labels)
        auc = roc_auc_score(predictions>=threshold,test_Labels)
        print("Test Accuracy:", acc)
        print("Recall Accuracy:", recall)
        print("AUC Score:", auc)

        perf["model"].append("ActiModel")
        perf["win(sec)"].append(winmin*60)
        perf["acc"].append(acc)
        perf["recall"].append(recall)
        perf["auc"].append(auc)
        model_ls.append(model)
        hist_ls.append(H)
    perf_df = pd.DataFrame(perf)
    print(perf_df)
    return perf_df, model_ls, hist_ls
    
