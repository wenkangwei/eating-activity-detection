
from packages import *
from collections import defaultdict

def weight_accuracy(y_true, y_pred,weight, print_flag=True):
    TP = sum( (y_true==1) &(y_pred==1) )
    FN = sum( (y_true==1) &(y_pred==0) )
    TN = sum( (y_true==0) &(y_pred==0) )
    FP = sum( (y_true==0) &(y_pred==1) )
    if print_flag:
        print("TP: ",TP, "FP: ",FP, "TN: ",TN, "FN: ",FN)
    return (weight*TP + TN)/(weight*(TP+FN) + (TN+FP))

def test_models_time_metric(balance_ratio,winmin=1, stridesec = 5,names= ["wenkanw"],random_seed=1000, split_day=False, test_balance=False, test_CAD=False):
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
                wacc2 = weight_accuracy(test_Labels,prediction, weight=balance_ratio)
            
            
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



def print_time_metrics(result, old_result = None,round_decimal = 3,):
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
    
    if not isinstance(old_result, type(None)):
        # remove average performance from old results and append new data to table
        drop_vals = result["dataset"].values.tolist()
        drop_vals.append("average performance")
        idx = []
        for i in range(len(old_result)):
            if old_result['dataset'].iloc[i] in drop_vals:
                idx.append(i)
                
        #idx =old_result[(old_result['dataset']=="average performance") | (old_result['dataset']==result["dataset"].values[0])].index
        print("index",idx)
        new_result = old_result.drop(index=idx,axis=0)
        perf_df= new_result.append(perf_df, ignore_index=True)
        
    mean_perf = pd.DataFrame(columns = perf_df.keys())
    mean_perf = mean_perf.append({"dataset":"average performance","win(sec)":"-"},ignore_index=True)
    for key in perf_df.keys():
        if key.lower() != "dataset" and  key.lower() != "win(sec)":
            mean_perf[key].at[0] = perf_df[key].mean().round(round_decimal)
    
    perf_df = perf_df.append(mean_perf,ignore_index=True)
            
    return perf_df








##########################
# Episode metric
##########################
hythreshold = {'wenkanw':[0.8, 0.3], 'adam':[0.8,0.3],'lawler':[0.8,0.3], 'shaurya':[0.8,0.3]}
def test_models_episode_metric(winmin=6, stridesec = 5,names= ["wenkanw"],random_seed=1000,
                               test_balance=False, test_CAD=False,
                               test_alldata=False,threshold= hythreshold,
                               load_proba_flag=True, use_group_threshold = 0,
                              proba_path ="../results/possibility_results/"):
    perf = defaultdict(list)
    meal_info = defaultdict(list)
    
    # Don't test on CAD test set if it is not enabled
    if not test_CAD and "CAD" in names:
        names.remove("CAD")
    group_threshold = {'wenkanw':[0.8, 0.3], 'adam':[0.8,0.3],'lawler':[0.8,0.3], 'shaurya':[0.8,0.3]}
    for name in names:
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
        proba_path += person+"/"
        for i in range(len(models["suffix"])):
            suffix = models["suffix"][i]
            model = models["model"][i]
            # if the dataset is CAD group dataset and model is individual model
            # we don't need to make prediction on that data
            path= proba_path
            if use_group_threshold==0:
                high_th, low_th = threshold[name][0], threshold[name][1]
            elif use_group_threshold==1:
                if suffix == "GroupModel":
                    high_th, low_th = group_threshold[name][0], group_threshold[name][1]
                else:
                    high_th, low_th = threshold[name][0], threshold[name][1]
            else:
                high_th, low_th = group_threshold[name][0], group_threshold[name][1]
                
            
            if suffix == "GroupModel":
                path = proba_path +"group_"
                
                
            result = hysteresis_threshold(model, meal_data,start_threshold=high_th, end_threshold=low_th,
                                          winmin = 6, stepsec=5, episode_min = 1.,
                                         load_proba_flag=load_proba_flag, path =path)
            episode_perf_df = get_episode_metrics(result,meal_data)
            perf["TPR: "+suffix].append(episode_perf_df["TPR"].iloc[0])
            perf["FP/TP: "+suffix].append(episode_perf_df["FP/TP"].iloc[0])
            perf["TP: "+suffix].append(episode_perf_df["TP"].iloc[0])
            perf["FP: "+suffix].append(episode_perf_df["FP"].iloc[0])
            perf["FN: "+suffix].append(episode_perf_df["FN"].iloc[0])
            
            print(episode_perf_df)

    perf_df = pd.DataFrame(perf)
    return perf_df




def print_episode_metrics(result,old_result= None, round_decimal=3):
    """
    print the episode_perf_df result from test_models_episode_metric
    in suitable order
    """
    result_df= result[['dataset','Days' ,'Meal_Hours',"Meal_Counts","win(sec)",
            "TPR: Individual-Model","TPR: GroupModel","FP/TP: Individual-Model","FP/TP: GroupModel",
           "TP: Individual-Model","TP: GroupModel",
           "FP: Individual-Model","FP: GroupModel",
           "FN: Individual-Model","FN: GroupModel"]]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']   
    for key in result_df.columns:
        if result_df[key].dtype in numerics:
            #result_df[key] = np.round(result_df[key].values, round_decimal) 
            for i in range(len(result_df[key])):
                result_df[key].at[i] = result_df[key].iloc[i].round(round_decimal)
            
    if not isinstance(old_result, type(None)):
        # remove average performance from old results and append new data to table
        drop_vals = result["dataset"].values.tolist()
        drop_vals.append("average performance")
        idx = []
        for i in range(len(old_result)):
            if old_result['dataset'].iloc[i] in drop_vals:
                idx.append(i)
        
        #idx =old_result[(old_result['dataset']=="average performance") | (old_result['dataset'].values == result["dataset"].values)].index
        print("index",idx)
        new_result = old_result.drop(index=idx,axis=0)
        result_df= new_result.append(result_df, ignore_index=True)
    
    mean_perf = pd.DataFrame(columns = result_df.keys())
    mean_perf = mean_perf.append({"dataset":"average performance","win(sec)":"-",'Days':"-" ,
                                  'Meal_Hours':"-","Meal_Counts":"-"},ignore_index=True)
    for key in mean_perf.keys():
        if key not in ['dataset','Days' ,'Meal_Hours',"Meal_Counts","win(sec)"]:
            mean_perf[key].at[0] = result_df[key].mean().round(round_decimal)
    
    result_df = result_df.append(mean_perf,ignore_index=True)
    return result_df
