
from packages import *
from hysteresis_threshold import *
from dataset import *
from collections import defaultdict
from metrics import *

def cross_validation_metrics(model_arch,balance_ratio,names= ["wenkanw"], fold_num = 5, winmin=6, stridesec = 5,model_name= "acti_model",epochs = 20,
                                 random_seed=1000, split_day=False, test_balance=False, re_train = False,load_data=True,
                                 test_CAD=False, metrics =['time','episode'], ind_threshold= None,load_proba_flag=True):
    import os
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_score
    batch_size = 128
    time_perf = defaultdict(list)
    episode_perf = defaultdict(list)
    meal_info = defaultdict(list)
    time_individual_perf = defaultdict(list)
    episode_individual_perf = defaultdict(list)
    time_group_perf = defaultdict(list)
    episode_group_perf = defaultdict(list)
    
    group_threshold = [0.8,0.3] #{'wenkanw':, 'adam':[0.8,0.3],'lawler':[0.8,0.3], 'shaurya':[0.8,0.3]}
                
    model = None
    kf = KFold(n_splits=5, random_state= 1000,shuffle=False)
    # Don't test on CAD test set if it is not enabled
    if not test_CAD and "CAD" in names:
        names.remove("CAD")
    
    
    for name in names:
        #load individual whole dataset
        person = name
        meal_data = Person_MealsDataset(person_name= person, file_name = "all_files_list", winmin = winmin,stridesec = stridesec)
        samples,labels =  meal_data.data_indices, meal_data.labels
            
        if name !="CAD":
            meal_counts, min_counts,hour_counts, day_counts,total_hours = meal_data.get_mealdataset_info(person_name=name, file_ls_doc="all_files_list.txt")
        else:
            # data from paper
            meal_counts, min_counts,hour_counts, day_counts,total_hours = 1063, 250*60,250, 354, 4680
        meal_info["dataset"].append(name)
        meal_info["Days"].append(day_counts)
        meal_info["Meal_Hours"].append(round(hour_counts,1)) 
        meal_info["Meal_Counts"].append(meal_counts) 
        meal_info["Total_Hours"].append(total_hours) 
        
        time_perf["dataset"].append(name)
        time_perf["win(sec)"].append(winmin*60)
        
        episode_perf["dataset"].append(name)
        episode_perf["win(sec)"].append(winmin*60)
        
        days = np.unique(meal_data.data_indices[:,0])
        samples,labels =  meal_data.data_indices, meal_data.labels
#         samples,labels = meal_data.get_subset([i for i in range(len(meal_data.labels))])

        # K-fold cross validation
        for fold, (day_train_idx, day_test_idx) in enumerate(kf.split(days)):
            print("Fold: %d"%(fold),"Train on days: ",day_train_idx, "Test on days: ",day_test_idx)
            day_train_idx = day_train_idx.tolist()
            day_test_idx = day_test_idx.tolist()
            train_indices = []
            test_indices = []
            # partition dataset by days
            for i, day in enumerate(meal_data.data_indices[:,0]):
                if day in day_train_idx:
                    train_indices.append(i)
                else:
                    test_indices.append(i)
#             print("Train indices: ", train_indices)
#             print("Test indices: ", test_indices)
#             assert False
            # balance train set
            trainset_labels = labels[train_indices]
            train_indices_balanced = balance_data_indices(trainset_labels,data_indices= train_indices,mode="under", shuffle=True,random_state = random_seed,replace= False)
            
            testset_labels = labels[test_indices]
            if test_balance:
                #balance test set
                test_indices = balance_data_indices(testset_labels,data_indices= test_indices,mode="under", shuffle=True,random_state = random_seed,replace= False)
            else:
                test_indices = test_indices 
                   
            
            # split validation set
            balanced_trainset_labels = labels[train_indices_balanced]
            train_indices, valid_indices = split_train_test_indices(X= train_indices_balanced,
                                                                    y = balanced_trainset_labels, test_size = 0.2,
                                                                   random_seed = random_seed)
                    
            if not load_data and "time" not in metrics:
                balancedData, balancedLabels = meal_data.get_subset([])
                valid_balancedData, valid_balancedLabels = meal_data.get_subset([])
                test_Data, test_Labels = meal_data.get_subset([])
            else:
                # Get numpy dataset: balanced trainset, validation set, test set
                balancedData, balancedLabels = meal_data.get_subset(train_indices)
                valid_balancedData, valid_balancedLabels = meal_data.get_subset(valid_indices)
                test_Data, test_Labels = meal_data.get_subset(test_indices)

                # balancedData, balancedLabels = samples[train_indices],labels[train_indices]  
                # valid_balancedData, valid_balancedLabels = samples[valid_indices],labels[valid_indices] 
                # test_Data, test_Labels = samples[test_indices],labels[test_indices]

                print("Train on : ", sum(balancedLabels==1),"positive samples, ",sum(balancedLabels==0)," negative samples" )
                print("Testing on : ", sum(valid_balancedLabels==1),"positive samples, ",sum(valid_balancedLabels==0)," negative samples" )
                print("Testing on : ", sum(test_Labels==1),"positive samples, ",sum(test_Labels==0)," negative samples" )
            
            
            #train models
            pathtemp = "../models/" + name+"_models" +"/"+"cv_fold_"+str(fold) +"_"+model_name+"_M_F_"
            modelpath = pathtemp + "{:f}Min.h5".format(winmin)
            jsonpath = pathtemp + "{:f}Min.json".format(winmin)
            
            # if model doesn't exist or re_train is enabled, then re_trian
            # otherwise, just load model
            if not os.path.isfile(modelpath) or re_train:
                #training settings
                win_size = 15*winmin*60
                model =model_arch(input_shape =(win_size,6) )
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                
                mcp_save = tf.keras.callbacks.ModelCheckpoint(modelpath, save_best_only=True, monitor='val_accuracy')
                scheduler = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=3, verbose=0,
                                                     mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.)
                ##########train model ###############
                H = model.fit(x=balancedData, y = balancedLabels,
                               validation_data=(valid_balancedData, valid_balancedLabels),
                            epochs = epochs, batch_size=batch_size, verbose=1,
                            callbacks=[mcp_save,scheduler]) # removed addons.LossHistory(jsonpath) for compatibility with TensorFlow 2.2.0, needs to be re-added at some point

                print("Max value: ", max(H.history['accuracy']), " at epoch", np.argmax(H.history['accuracy']) + 1)
                print("Model saved to path: ",modelpath)
            else:
                model  = tf.keras.models.load_model(modelpath)
                
            # obtain individual model and pre-trained group model
            individual_model = model
            group_model_W  = tf.keras.models.load_model('../models/CAD_models/acti_6min_M_F_6.000000Min.h5')
            models = {"suffix":['Individual-Model','GroupModel'],  "model":[individual_model,group_model_W]}
            
            ##### Test models####
            ############ Time metrics ##########
            if 'time' in metrics:
                for i in range(len(models["suffix"])):
                    suffix = models["suffix"][i]
                    model = models["model"][i]
                    # if the dataset is CAD group dataset and model is individual model
                    # we don't need to make prediction on that data
                    if name == "CAD" and suffix =='Individual-Model':
                        acc = None
                        auc = None
                        recall = None
                    else:

                        predictions = model.predict(x=test_Data).squeeze(1)
                        threshold = 0.5
                        prediction = (predictions>=threshold).astype(int)
                        wacc =  balanced_accuracy_score(test_Labels,prediction)
                        acc =  accuracy_score(test_Labels,prediction)
                        recall = recall_score(test_Labels,prediction)
                        f1 = f1_score(test_Labels,prediction)
                        precision = precision_score(test_Labels,prediction)
                        # weighted accuracy 2 is computed by (weight*TP +TN)/(weight*(TP+FN) + (TN+FP))
                        wacc2 = weight_accuracy(test_Labels,prediction,weight = balance_ratio,print_flag=False)
                        
                        # store performance for one fold 
                        if suffix == "GroupModel":
                            time_group_perf["WAcc: "+suffix].append(wacc)
                            time_group_perf["WAcc2: "+suffix].append(wacc2)
                            time_group_perf["Recall: "+suffix].append(recall)
                            time_group_perf["Precision: "+suffix].append(precision)
                            time_group_perf["F1: "+suffix].append(f1)
                            time_group_perf["Acc: "+suffix].append(acc)
                        else:
                            time_individual_perf["WAcc: "+suffix].append(wacc)
                            time_individual_perf["WAcc2: "+suffix].append(wacc2)
                            time_individual_perf["Recall: "+suffix].append(recall)
                            time_individual_perf["Precision: "+suffix].append(precision)
                            time_individual_perf["F1: "+suffix].append(f1)
                            time_individual_perf["Acc: "+suffix].append(acc)
                            

            ######## episode metric ############
            if "episode" in metrics:
                proba_path ="../results/possibility_results/"+person +"/cv_fold_"+str(fold)+"_"
                for i in range(len(models["suffix"])):
                    suffix = models["suffix"][i]
                    model = models["model"][i]
                    result_path = proba_path
                    if suffix == "GroupModel":
                        result_path += "group_"
                        high_th, low_th  = group_threshold[0],group_threshold[1]
                    else:
                        if ind_threshold:
                            high_th, low_th  = ind_threshold[name][0],ind_threshold[name][1] 
                        else:
                            high_th, low_th  = group_threshold[0],group_threshold[1]
                        
                    result = hysteresis_threshold(model, meal_data,days_ls = day_test_idx,start_threshold=high_th, end_threshold=low_th,
                                                  winmin = winmin, stepsec=stridesec, episode_min = 1.,
                                                 load_proba_flag=load_proba_flag, path =result_path)
                    episode_perf_df = get_episode_metrics(result,meal_data,days_ls = day_test_idx)
                    
                    TP = episode_perf_df["TP"].iloc[0]
                    FP = episode_perf_df["FP"].iloc[0]
                    FN = episode_perf_df["FN"].iloc[0]
                    if suffix == "GroupModel":
                        episode_group_perf["TP: "+suffix].append(TP)
                        episode_group_perf["FP: "+suffix].append(FP)
                        episode_group_perf["FN: "+suffix].append(FN)
                    else:
                        episode_individual_perf["TP: "+suffix].append(TP)
                        episode_individual_perf["FP: "+suffix].append(FP)
                        episode_individual_perf["FN: "+suffix].append(FN)

        if 'time' in metrics:
            for key in  time_group_perf.keys():
                time_perf[key].append(  np.mean(time_group_perf[key])) 
                time_group_perf[key].clear()
                
            for key in  time_individual_perf.keys():
                time_perf[key].append(  np.mean(time_individual_perf[key])) 
                time_individual_perf[key].clear()
                
        if 'episode' in metrics:
            for key in episode_group_perf.keys():
                episode_perf[key].append(  np.sum(episode_group_perf[key])) 
                episode_group_perf[key].clear()
            for key in episode_individual_perf.keys():
                episode_perf[key].append(  np.sum(episode_individual_perf[key])) 
                episode_individual_perf[key].clear()
            
    meal_info = pd.DataFrame(meal_info)
    episode_perf = pd.DataFrame(episode_perf)
    time_perf = pd.DataFrame(time_perf)
    # Compute TPR, FP/TP for all models
    for suffix in ['Individual-Model','GroupModel']:
            episode_perf["TPR: "+suffix] = episode_perf['TP: '+suffix]/(episode_perf['TP: '+suffix] + episode_perf['FN: '+suffix])
            episode_perf['FP/TP: '+suffix] = episode_perf['FP: '+suffix]/episode_perf['TP: '+suffix]
            
    return meal_info, time_perf,episode_perf





def test_threshold_cv(datasets ,ts_ls=[],te_ls=[], fold_num=5,round_num = 3,path_name = "../results/possibility_results/"):
    """
    Test the hysteresis threshold values  for individual models based on generated possibility in csv files from hysteresis_threshold function
    
    datasets:  a dictionary of datasets generated from create_dataset() in dataset.py
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, random_state= 1000,shuffle=False)
    res = pd.DataFrame()
    
    for person in datasets.keys():
        data = datasets[person]
        for ts in ts_ls:
            for te in te_ls:
                TP,FN, FP = 0,0,0
                df= {'dataset':[],'Ts':[],"Te":[],'TP':[],'FN':[],"FP":[],'TPR':[],'FP/TP':[]}
                days = np.unique(data.data_indices[:,0])
                samples,labels =  data.data_indices,data.labels
                # K-fold cross validation
                for fold, (day_train_idx, day_test_idx) in enumerate(kf.split(days)):
                    day_train_idx, day_test_idx =day_train_idx.tolist(), day_test_idx.tolist()
                    result_path =path_name +person+"/cv_fold_"+str(fold)+"_"
                    model_name = '../models/{}_models/cv_fold_{}_acti_model_M_F_6.000000Min.h5'.format(person, fold)
                    model = tf.keras.models.load_model(model_name)
                    ht_result = hysteresis_threshold(model, data,days_ls= day_test_idx,start_threshold=ts, end_threshold=te, 
                                  winmin = 6, stepsec=5, episode_min = 1., load_proba_flag=True,path =result_path)
                    episode_perf = get_episode_metrics(ht_result,data,days_ls= day_test_idx)
                    TP += episode_perf["TP"].iloc[0]
                    FP += episode_perf["FP"].iloc[0]
                    FN += episode_perf["FN"].iloc[0]
                    
                df["dataset"].append(person)
                df["Ts"].append(ts)
                df["Te"].append(te)
                df["TP"].append(TP)
                df["FN"].append(FN)
                df["FP"].append(FP)
                df["TPR"].append(round(TP/(TP+FN) if (TP+FN)>0 else 0,round_num))
                df["FP/TP"].append(round(FP/TP, round_num))
                df = pd.DataFrame(df)
                print(df)
                res = res.append(df, ignore_index=True)
                
    return res  

def find_optimal_threshold(threshold_results,mode="min_fp", min_tpr= 0.85, max_fp=1.):
    """
    To find the optimal threshold for each individual model, based on threshold_results from  function test_threshold_cv()
    
    threshold_results: output dataframe from test_threshold_cv() function
    """
    best_threshold = pd.DataFrame()
    threshold_results["ratio"] = threshold_results["TP"].values/ (threshold_results["TP"].values+threshold_results["FN"].values +threshold_results["FP"].values)
    for person in threshold_results["dataset"].unique():
        
        df= threshold_results.loc[threshold_results['dataset']==person]
        if mode =="min_fp":
            # find min FP/TP with TPR inside range
            df= df.loc[threshold_results["TPR"]>min_tpr]
            if len(df) == 0:
                df= df.iloc[threshold_results["TPR"].argmax()]
            else:
                df = df.iloc[df["FP/TP"].argmin()]
        elif mode == "max_tpr":
            # find max TPR with FP/TP inside range
            df = df.loc[df["FP/TP"]<max_fp]
            
            if len(df) == 0:
                df= df.iloc[threshold_results["FP/TP"].argmin()]
            else:
                df= df.iloc[threshold_results["TPR"].argmax()]
        else:
            df= df.loc[threshold_results["TPR"]>min_tpr]
            if len(df) == 0:
                df= df.iloc[threshold_results["TPR"].argmax()]
            else:
                df = df.iloc[df["ratio"].argmax()]
                
        best_threshold = best_threshold.append(df)
        thresholds = {}
        for name in best_threshold['dataset'].values:
            thresholds[name] = [ best_threshold[best_threshold['dataset']==name]['Ts'].values[0] ,
                                best_threshold[best_threshold['dataset']==name]['Te'].values[0] ]
        #best_threshold[["dataset","Ts","Te"]]
    best_threshold = best_threshold[["dataset","Ts","Te","TPR","FP/TP","TP","FP","FN"]]
    return best_threshold , thresholds


    
