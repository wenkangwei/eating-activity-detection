
import numpy as np
import pandas as pd
import torch
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from data_loader import loadEvents
def get_meal_info(person_name = None,file_ls = [], file_ls_doc=None,root_path = "../data/",print_file=False,round_decimal=1):
        """
        if file_ls is not given, then get file_ls according to person_name
        file path = root_path + file name in all_files_list.txt

        return:
            meal event count, total minutes of all meals, total hours of all meals,total day counts

        """
        meal_counts = 0
        sec_counts = 0
        min_counts = 0
        hour_counts = 0
        total_hours = 0
        total_mins = 0
        total_sec = 0
        day_counts = 0 
        if person_name ==None:
            return meal_counts, min_counts,hour_counts, day_counts, total_hours

        data_indices_file = "../data-file-indices/" +person_name+"/all_files_list.txt"
        fp = open(data_indices_file,"r")
        txt = fp.read()
        fp.close()
        file_ls = txt.split("\n")
        while '' in file_ls:
            file_ls.remove('')
       
        day_counts = len(file_ls)
        
        for file_name in file_ls:
            file_name = root_path + file_name
            TotalEvents, EventStart, EventEnd, EventNames, TimeOffset,EndTime = loadEvents(file_name, debug_flag = False, print_file=print_file)
            meal_counts += TotalEvents
            total_sec +=  abs(EndTime - TimeOffset)
            for i in range(len(EventStart)):
                sec_counts += ( EventEnd[i]- EventStart[i])//15
                
        total_hours = total_sec/(60*60)
        min_counts = sec_counts/60
        hour_counts = min_counts/60
        average_meal_per_day = meal_counts/len(file_ls)
        average_hour_per_meal = hour_counts/meal_counts
        # round numbers
        total_hours = round(total_hours, round_decimal)
        min_counts = round(min_counts, round_decimal)
        hour_counts = round(hour_counts, round_decimal)
        average_meal_per_day = round(average_meal_per_day,round_decimal)
        average_hour_per_meal = round(average_hour_per_meal, round_decimal)
        
        no_eating_hours = total_hours - hour_counts
        weight_ratio = round(no_eating_hours/hour_counts, round_decimal)
        result = pd.DataFrame({"dataset": person_name,"Days":day_counts, 
                      "Total Hours":total_hours,"Meal Counts":meal_counts,
                      "Average Meal Counts Per Day":average_meal_per_day,"Average Hours Per Meal": average_hour_per_meal,
                      "Eating Hours":hour_counts, "No Eating Hours":no_eating_hours,
                     "Balance Ratio(no_eat/eat)":weight_ratio},index=[0])
    
        return result

          
def get_dataset_info(names= ["wenkanw"],winmin=6,stridesec=5):
    """
    Function to get information of meal dataset
    """
    meal_info = defaultdict(list)
    dataset_results = pd.DataFrame()
    for name in names:
        result = get_meal_info(person_name=name)
        if dataset_results.empty:
            dataset_results = result
        else:
            dataset_results = dataset_results.append(result,ignore_index=True)
    
    # append total summary
#     print( dataset_results)
    total_result=pd.DataFrame({"dataset":"total"},columns = dataset_results.columns,index=[0])
    # append average summary
    average_result=pd.DataFrame({"dataset":"average"},columns = dataset_results.columns,index=[0])
    key_ls = ["Days","Total Hours","Meal Counts","Eating Hours","No Eating Hours"]
    for key in dataset_info.columns:
        if key in key_ls:
            total_result[key].at[0] = round(dataset_results[key].sum() ,1)
            average_result[key].at[0] = round(dataset_results[key].mean(),1)

    ls = [total_result, average_result]
    for df in ls:
        df["Average Meal Counts Per Day"].at[0] = round(df["Meal Counts"].values[0]/df["Days"].values[0], 1)
        df["Average Hours Per Meal"].at[0] =round( df["Eating Hours"].values[0]/df["Meal Counts"].values[0], 1)
        df["Balance Ratio(no_eat/eat)"].at[0] =round(df["No Eating Hours"].values[0]/df["Eating Hours"].values[0],1)
        dataset_results =dataset_results.append(df,ignore_index=True)

    return dataset_results







from dataset import create_train_test_file_list,  balance_data_indices  #Person_MealsDataset,
from utils import *
from model import *
def train_models(model, win_ls = [],EPOCHS = 10,stridesec = 5,name = "wenkanw",model_name="acti_6min" ,
                 random_seed= 1000, split_day=False,test_balanced=False,
                create_file_ls = False):
    """
    Train model using train/test spit
    """
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





from collections import defaultdict
name_ls = ["wenkanw",'adam',"lawler","shaurya"]
dataset_info = get_dataset_info(names= name_ls)
balance_ratio = dataset_info[dataset_info["dataset"]=="total"]['Balance Ratio(no_eat/eat)'].values[0]

def weight_accuracy(y_true, y_pred,weight= balance_ratio):
    TP = sum( (y_true==1) &(y_pred==1) )
    FN = sum( (y_true==1) &(y_pred==0) )
    TN = sum( (y_true==0) &(y_pred==0) )
    FP = sum( (y_true==0) &(y_pred==1) )
    print("TP: ",TP, "FP: ",FP, "TN: ",TN, "FN: ",FN)
    return (weight*TP + TN)/(weight*(TP+FN) + (TN+FP))

def test_models_time_metric(winmin=1, stridesec = 5,names= ["wenkanw"],random_seed=1000, split_day=False, test_balance=False, test_CAD=False):
    """
    Test time metrics
    """
    perf = defaultdict(list)
    meal_info = defaultdict(list)
    # Don't test on CAD test set if it is not enabled
    if not test_CAD and "CAD" in names:
        names.remove("CAD")
        
    for name in names:
        person = name
        if split_day:
            meal_data = Person_MealsDataset(person_name= person, file_name = "test_files", winmin = winmin,stridesec = stridesec)

            # balance test set
            testset_labels = meal_data.labels
            if test_balance:
                test_indices = balance_data_indices(testset_labels,data_indices=[i for i in range(len(meal_data))] ,mode="under", shuffle=True,random_state = random_seed,replace= False)
            else:
                test_indices = [i for i in range(len(meal_data))]
            # get numpy dataset
            test_Data, test_Labels = meal_data.get_subset(test_indices)
        else:            
            meal_data = Person_MealsDataset(person_name= person, file_name = "all_files_list", winmin = winmin,stridesec = stridesec)
            samples,labels =  meal_data.data_indices, meal_data.labels
            # split train set and test set
            train_indices, test_indices = split_train_test_indices(X= [i for i in range(len(labels))],
                                                                            y = labels, test_size = 0.2,
                                                                           random_seed = random_seed)
            
            if test_balance:
                testset_labels = labels[test_indices]
                test_indices = balance_data_indices(testset_labels,data_indices= test_indices,mode="under", shuffle=True,random_state = random_seed,replace= False)
            else:
                test_indices = test_indices
            testset_labels = labels[test_indices]
            print("Testing on : ", sum(testset_labels==1),"positive samples, ",sum(testset_labels==0)," negative samples" )
            test_Data, test_Labels = meal_data.get_subset(test_indices)
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
        
        perf["dataset"].append(name)
        perf["win(sec)"].append(winmin*60)
        
            
        
        
        from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_score
        group_model_W  = tf.keras.models.load_model('../models/CAD_models/acti_6min_M_F_6.000000Min.h5')
        
        if name != "CAD":
            if split_day:
                individual_model = tf.keras.models.load_model('../models/'+ name+ '_models/acti_6min_split_day_M_F_6.000000Min.h5')
            else:    
                individual_model = tf.keras.models.load_model('../models/'+ name+ '_models/acti_6min_M_F_6.000000Min.h5')
        models = {"suffix":['Individual-Model','GroupModel'],  "model":[individual_model,group_model_W]}
        
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
                print("Test label: ",test_Labels)
                print("Predictions:",prediction)
                
                # weighted accuracy 2 is computed by (weight*TP +TN)/(weight*(TP+FN) + (TN+FP))
                wacc2 = weight_accuracy(test_Labels,prediction)
            
            
            print("Weighted Accuracy:", wacc)
            print("Weighted Accuracy2:", wacc2)
            print("Recall:", recall)
            print("Precision:", precision)
            print("F1:", f1)
            print("Test Accuracy:", acc)
            
            perf["WAcc: "+suffix].append(wacc)
            perf["WAcc2: "+suffix].append(wacc2)
            perf["Recall: "+suffix].append(recall)
            perf["Precision: "+suffix].append(precision)
            perf["F1: "+suffix].append(f1)
            perf["Acc: "+suffix].append(acc)

    meal_info = pd.DataFrame(meal_info)
    perf_df = pd.DataFrame(perf)
    return meal_info, perf_df


def print_time_metrics(result, round_decimal = 3):
    perf_df = pd.DataFrame()
    mykeys = ["dataset","win(sec)","WAcc", "F1","Precision","Recall"]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']    
    for k in mykeys:
        for key in result.keys():
            if k in key:
                perf_df[key] = result[key]
                if result[key].dtype in numerics:
                    perf_df[key] = np.round(perf_df[key],round_decimal)
    for key in result.keys():
        if "WAcc" not in key and "Acc" in key:
            perf_df[key] = result[key].values.round(round_decimal)
    
    mean_perf = pd.DataFrame(columns = perf_df.keys())
    mean_perf = mean_perf.append({"dataset":"average performance","win(sec)":"-"},ignore_index=True)
    for key in perf_df.keys():
        if key.lower() != "dataset" and  key.lower() != "win(sec)":
            mean_perf[key].at[0] = perf_df[key].mean().round(round_decimal)
    
    perf_df = perf_df.append(mean_perf,ignore_index=True)
            
    return perf_df



def hysteresis_threshold(model, data,start_threshold=0.8, end_threshold=0.3, winmin = 6,
                        stepsec=5, episode_min = 1.,t_pause = 900):
    """
    model: tensorflow model
    data:  This dataset must be the self-defined class of Person_MealsDataset  datasetset in my dataset.py/pytorch dataset without using shuffle. 
    Keep the order of dataset after extracting window samples!  You can also define your own dataset using class object to create the interface
    
    start_threshold: the high threshold of the beginning of segmentation
    
    end_threshold: the end threshold of the end of segmentation
    
    winmin: size of a window sample in unit of  minute
    
    stepsec: stride to move the window in unit of second / the number of second between two adjacent window samples
    
    episode_min: the minimum length of eating episode in unit of minute. If end of segmentation -start of segmentation < episode_min,
        then the episode will not be counted
    
    """
    result_ls = []
    
    
    days = set(data.data_indices[:,0])
    for day in days:
        # Select and Extract the data and labels of the corresponding day from the whole dataset
        sample_indices= np.where(data.data_indices[:,0]==day)[0]
        result = {'day':day,"stepsec": stepsec,'segment_start':[], 'segment_end':[],'proba':[],'predictions':np.zeros([len(sample_indices)]),'labels':[],"segment_count":0}
        
        # get the numpy array of samples and labels
        samples, labels = data.get_subset(sample_indices)
        probas = model(samples)
        state = 0
        start = 0
        end = 0 
        pause_counter = 0
        # one day data
        print("Day: ",day)
        for i in range(len(sample_indices)):
            #print("i:",i)
            #sample, label = data[i][0].numpy(),data[i][1]
            #sample = np.expand_dims(sample,axis=0)
            #proba = model(sample).numpy()[0][0]
            sample = samples[i]
            label = labels[i]
            proba = probas[i].numpy()[0]
            
            result['proba'].append(proba)
            result['labels'].append(label)
            
            if state ==0 and proba > start_threshold:
                state = 1
                start = i
            elif state == 1 and proba <end_threshold:
                state = 2
                end = i+1
                pause_counter = 0
            elif state ==2:
                if proba > start_threshold:
                    state = 1
                else:
                    pause_counter += stepsec
                    if pause_counter >= t_pause:
                        # convert time to second and check threshold
                        if (end-start)*stepsec >= episode_min*60:
                            # save data
                            result['segment_start'].append(start)
                            result['segment_end'].append(end)
                            result['segment_count'] += 1
                            result['predictions'][start:end] = 1
                            pass
                        end = 0
                        state = 0
        if state != 0:
            # if segment ended at the end of data
            if end != 0:
                result['segment_start'].append(start)
                result['segment_end'].append(end)
                result['predictions'][start:end] = 1
            else:
                result['segment_count'] -= 1  
            result['segment_count'] += 1
            
        result_ls.append(result)
        print("Segmentation Completed. ")
                            
    return pd.DataFrame(result_ls)




def get_episode_metrics(result, meal_data):
    """
    result: result from  hysteresis threshold function
    meal_data: meal dataset of Person_MealData
    
    """
    
    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
    total_preds = []
    total_labels = []
    perf = {"TPR":[],"FP/TP":[],"TP":[], "FP":[],"FN":[]}
    tpr = 0. 
    FP_TP = 0.
    TP = 0
    FP = 0
    FN = 0
    # get segmentation of ground truth labeled by user
    start_ls, end_ls = meal_data.get_GT_segment()
    
    meal_counts, min_counts,hour_counts, day_counts,total_hours = meal_data.get_mealdataset_info()
    
    # iterate every day
    for i in range(len(result)):
        #preds = result.iloc[i]['predictions']
        #labels =  result.iloc[i]['labels']
        event_start, event_end= start_ls[i], end_ls[i]
        detect_start, detect_end = result.iloc[i]['segment_start'],result.iloc[i]['segment_end']
        GT = np.array([-1]*len(event_start) )  # default all meals are missing -1, FN
        detect = np.array([-1]*len(detect_start)) # default all detected meals are wrong -1, FP
        for index in range(len(event_start)):
            # e_s: event start,  e_e: event end
            # d_s: detection start,  d_e: detection end
            e_s, e_e = event_start[index], event_end[index]
            for index2 in range(len(detect_start)):
                # convert segment from sec to index of data point
                d_s = detect_start[index2] * result.iloc[i]['stepsec']*15
                d_e = detect_end[index2]* result.iloc[i]['stepsec']*15
                #print("ds: {} d_e: {}, e_s:{}, e_e: {}".format(d_s,d_e, e_s, e_e))
                if (e_s>=d_s and e_s <= d_e) or (d_s>= e_s and d_s<= e_e):
                    GT[index] = index2
                    detect[index2] = index
        #print("GT:",GT, "Detect:", detect)
        TP += sum(GT!=-1)
        FN += sum(GT==-1)
        FP += sum(detect==-1)
                
    
    print("total_meal:",meal_counts, "TP: ", TP, "FP: ", FP, "FN: ", FN)
    perf['TPR'].append(TP/(TP+FN))
    if TP ==0:
        perf['FP/TP'].append(None)
    else:
        perf['FP/TP'].append(FP/TP)
    perf["TP"].append(TP)
    perf["FP"].append(FP)
    perf["FN"].append(FN)
    result_df = pd.DataFrame(perf)
        
    return pd.DataFrame(result_df)


from collections import defaultdict

def test_models_episode_metric(winmin=6, stridesec = 5,names= ["wenkanw"],random_seed=1000, test_balance=False, test_CAD=False,test_alldata=False):
    perf = defaultdict(list)
    meal_info = defaultdict(list)
    threshold = {'wenkanw':[0.8, 0.3], 'adam':[0.8,0.3],'lawler':[0.8,0.3], 'shaurya':[0.8,0.3]}
    # Don't test on CAD test set if it is not enabled
    if not test_CAD and "CAD" in names:
        names.remove("CAD")
        
    for name in names:
        
        high_th, low_th = threshold[name][0], threshold[name][1]
        
        person = name
        # test episode metrics that split dataset by days
        if not test_alldata:
            meal_data = Person_MealsDataset(person_name= person, file_name = "test_files", winmin = winmin,stridesec = stridesec)
        else:
            meal_data = Person_MealsDataset(person_name= person, file_name = "all_files_list", winmin = winmin,stridesec = stridesec)
        # balance test set    
        if test_balance:
            testset_labels = meal_data.labels
            test_indices = balance_data_indices(testset_labels,data_indices=[i for i in range(len(meal_data))] ,mode="under", shuffle=True,random_state = random_seed,replace= False)
        else:
            test_indices = [i for i in range(len(meal_data))]
        # get numpy dataset
        #test_Data, test_Labels = meal_data.get_subset(test_indices)
        
        
        meal_counts, min_counts,hour_counts, day_counts,total_hours = meal_data.get_mealdataset_info(person_name=name)
                
        
        perf["Days"].append(day_counts)
        perf["Meal_Hours"].append(round(hour_counts,1)) 
        perf["Meal_Counts"].append(meal_counts) 
        perf["dataset"].append(name)
        perf["win(sec)"].append(winmin*60)
        
            
        from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_score
        group_model_W  = tf.keras.models.load_model('../models/CAD_models/acti_6min_M_F_6.000000Min.h5')
        
        individual_model = tf.keras.models.load_model('../models/'+ name+ '_models/acti_6min_split_day_M_F_6.000000Min.h5')
            
        models = {"suffix":['Individual-Model','GroupModel'],  "model":[individual_model,group_model_W]}
        
        for i in range(len(models["suffix"])):
            suffix = models["suffix"][i]
            model = models["model"][i]
            # if the dataset is CAD group dataset and model is individual model
            # we don't need to make prediction on that data
            
            result = hysteresis_threshold(model, meal_data,start_threshold=high_th, end_threshold=low_th, winmin = 6, stepsec=5, episode_min = 1.)
            episode_perf_df = get_episode_metrics(result,meal_data)
            perf["TPR: "+suffix].append(episode_perf_df["TPR"].iloc[0])
            perf["FP/TP: "+suffix].append(episode_perf_df["FP/TP"].iloc[0])
            perf["TP: "+suffix].append(episode_perf_df["TP"].iloc[0])
            perf["FP: "+suffix].append(episode_perf_df["FP"].iloc[0])
            perf["FN: "+suffix].append(episode_perf_df["FN"].iloc[0])
            
            print(episode_perf_df)

    perf_df = pd.DataFrame(perf)
    return perf_df



