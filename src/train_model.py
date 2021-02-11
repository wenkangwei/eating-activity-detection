
from packages import *
from dataset import *
from utils import *
from model import *
def train_models(model, win_ls = [],EPOCHS = 10,stridesec = 5,name = "wenkanw",model_name="acti_6min" ,
                 random_seed= 1000, split_day=False,test_balanced=False,
                create_file_ls = False):
    from numpy.random import seed
    seed(random_seed)
    random.seed(random_seed)
#     tf.set_random_seed(random_seed)
    from datetime  import datetime
    batch_size = 128
    outfile = sys.stdout
    perf = {"model":[],"data":[],"win(sec)":[], "wacc":[],"f1":[],"recall":[],"acc":[]}
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


        if split_day:
            pathtemp = "../models/" + name+"_models" +"/"+model_name+"_split_day_M_F_"
        else:
            pathtemp = "../models/" + name+"_models" +"/"+model_name+"_M_F_"
            
        #pathtemp = "../models/" + name +"/"+model_name+"_M_F_"
        modelpath = pathtemp + "{:f}Min.h5".format(winmin)
        jsonpath = pathtemp + "{:f}Min.json".format(winmin)
        
        print("Model to Save: ",modelpath)
        print()
        ########### Load the dataset################
        person = name
        if create_file_ls:
            create_train_test_file_list(file_name= "all_files_list.txt",person_name =name,
                         out_path = "../data-file-indices/",root_path= "../",
                         test_ratio = 0.2, print_flag = True, shuffle=True, random_state=random_seed)
        
        if split_day:
            

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
            if test_balanced:
                test_indices = balance_data_indices(testset_labels,data_indices=[i for i in range(len(meal_data_test))] ,mode="under", shuffle=True,random_state = random_seed,replace= False)
            else:
                # without balancing data
                test_indices = [i for i in range(len(meal_data_test))] 
                
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
            
            
            testset_labels = labels[test_indices]
            if test_balanced:
                #balance test set
                test_indices = balance_data_indices(testset_labels,data_indices= test_indices,mode="under", shuffle=True,random_state = random_seed,replace= False)
            else:
                test_indices = test_indices 
            
            
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

            # Get numpy dataset: balanced trainset, validation set, test set
            balancedData, balancedLabels = meal_data.get_subset(train_indices)
            valid_balancedData, valid_balancedLabels = meal_data.get_subset(valid_indices)
            test_Data, test_Labels = meal_data.get_subset(test_indices)
        

        #training settings
        mcp_save = tf.keras.callbacks.ModelCheckpoint(modelpath, save_best_only=True, monitor='accuracy')
        

        scheduler = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=3, verbose=0,
                                             mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.)
        
        ##########train model ###############
        H = model.fit(x=balancedData, y = balancedLabels,
                       validation_data=(valid_balancedData, valid_balancedLabels),
                    epochs = EPOCHS, batch_size=batch_size, verbose=1,
                    callbacks=[mcp_save,scheduler]) # removed addons.LossHistory(jsonpath) for compatibility with TensorFlow 2.2.0, needs to be re-added at some point

        print("Max value: ", max(H.history['accuracy']), " at epoch", np.argmax(H.history['accuracy']) + 1)

        from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, balanced_accuracy_score, f1_score
        predictions = model.predict(x=test_Data)
        threshold = 0.5
        wacc =  balanced_accuracy_score(test_Labels,predictions>=threshold)
        f1 =  f1_score(test_Labels,predictions>=threshold)
        acc =  accuracy_score(test_Labels,predictions>=threshold)
        recall = recall_score(test_Labels,predictions>=threshold)
        
        #auc = roc_auc_score(test_Labels,predictions>=threshold)
        print("Weighted Accuracy:", wacc)
        print("Test Accuracy:", acc)
        print("F1-score:", f1)
        print("Recall Accuracy:", recall)
        #print("AUC Score:", auc)

        perf["model"].append("ActiModel")
        perf["data"].append(name)
        perf["win(sec)"].append(winmin*60)
        perf["wacc"].append(wacc)
        perf["f1"].append(f1)
        perf["acc"].append(acc)
        perf["recall"].append(recall)
        #perf["auc"].append(auc)
        model_ls.append(model)
        hist_ls.append(H)
    perf_df = pd.DataFrame(perf)
    print(perf_df)
    return perf_df, model_ls, hist_ls


