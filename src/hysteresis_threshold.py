from packages import *
import os

def save_proba(result,path="proba.csv"):
    """
    Save predicted probability to csv files
    """
    df = pd.DataFrame(columns=["day","proba","labels"])
    for i, day in enumerate(result["day"].unique().tolist()):
        dat = {"day":[],"proba":[],"labels":[]}
        dat["proba"] = result["proba"].iloc[i]
        dat["day"] = [day]*len(result["proba"].iloc[i])
        dat["labels"] = result["labels"].iloc[i]
        dat = pd.DataFrame(dat)
        df =df.append(dat)
    df.to_csv(path, index=False)
    print("File "+path+" Saved")
    
def load_proba(path):
    """
    Load predicted probability on individual dataset
    """
    # key is day, value is possibility sequence
    proba_ls = {}
    labels_ls = {}
    df = pd.read_csv(path)
    for i in df['day'].unique():
        probas = df[df['day']==i]["proba"].values
        labels = df[df['day']==i]["labels"].values
        proba_ls[i] = probas
        labels_ls[i] = labels
    return proba_ls,labels_ls
    
def hysteresis_threshold(model, data,days_ls = [], start_threshold=0.8, end_threshold=0.3, winmin = 6,
                        stepsec=5, episode_min = 1.,t_pause = 900,load_proba_flag = True,
                         path ="../results/possibility_results/", file_name= None):
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
    file_name: csv file that store predicted possibility of model
    path: path to load / save predicted probability
    
    load_proba_flag: if enabled,  load saved probability to do hysteresis threshold
    """
    result_ls = []
    proba_ls,labels_ls = {}, {}
    if file_name == None:
        file_name =path+"{}_{}min_{}slide_proba.csv".format(data.person_name, winmin,stepsec)
    
    if load_proba_flag and  os.path.isfile(file_name):
        # load generated probability if we already generate it
        proba_ls,labels_ls = load_proba(file_name)
        pass
    else:
        # generate possibility for hysteresis threshold if we have not done yet
        if not days_ls:
            days = set(data.data_indices[:,0])
        else:
            days = days_ls
        #pbar = tqdm(days, total=len(days))
        for day in days:
            # Select and Extract the data and labels of the corresponding day from the whole dataset
            sample_indices= np.where(data.data_indices[:,0]==day)[0]

            # get the numpy array of samples and labels
            import time
            start_time = time.time()
            samples, labels = data.get_subset(sample_indices)
            #print("--- Get data:  %s seconds ---" % (time.time() - start_time))
            probas = model(samples).numpy().squeeze()
            #print("--- Prediction %s seconds ---" % (time.time() - start_time))
            print("--- Day %d: %s seconds ---" % (day, time.time() - start_time))
            proba_ls[day] = probas
            labels_ls[day] = labels
        df = {}
        df["day"] = list(days)
#         day_key = list(proba_ls.keys())
#         day_key.sort()
        df["proba"]= [proba_ls[k] for k in days]
        df['labels'] = [labels_ls[k] for k in days]
        df = pd.DataFrame(df)
        save_proba(df,path=file_name)   

    if not days_ls:
        days = set(data.data_indices[:,0])
    else:
        days = days_ls
    pbar = tqdm(days, total=len(days))
    for day in pbar:
        
        # Select and Extract the data and labels of the corresponding day from the whole dataset
        sample_indices= np.where(data.data_indices[:,0]==day)[0]
        
        probas = proba_ls[day]
        labels = labels_ls[day]
        result = {'day':day,"stepsec": stepsec,'segment_start':[], 'segment_end':[],'proba':[],'predictions':np.zeros([len(sample_indices)]),'labels':[],"segment_count":0}
        state = 0
        start = 0
        end = 0 
        pause_counter = 0
        # one day data
        #print("Day: ",day)
        for i in range(len(sample_indices)):
            #print("i:",i)
            #sample, label = data[i][0].numpy(),data[i][1]
            #sample = np.expand_dims(sample,axis=0)
            #proba = model(sample).numpy()[0][0]
            #sample = samples[i]
            label = labels[i]
            proba = probas[i]
            
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
#         print("--- One Day: %s seconds ---" % (time.time() - start_time))    
        result_ls.append(result)
    print("Segmentation Completed. ")
    result_ls = pd.DataFrame(result_ls)
                      
    return result_ls



def get_episode_metrics(result, meal_data,days_ls= None):
    """
    Obtain and format the episode metric results 
    
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
    if days_ls:
        start_ls = [start_ls[day] for day in days_ls]
        end_ls =  [end_ls[day] for day in days_ls]
    
    meal_counts, min_counts,hour_counts, day_counts,total_hours = meal_data.get_mealdataset_info()
    
    # iterate every day
    for i in range(len(start_ls)):
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
    perf['TPR'].append(TP/(TP+FN) if (TP+FN)>0 else 0)
    if TP ==0:
        perf['FP/TP'].append(None)
    else:
        perf['FP/TP'].append(FP/TP)
    perf["TP"].append(TP)
    perf["FP"].append(FP)
    perf["FN"].append(FN)
    result_df = pd.DataFrame(perf)
        
    return pd.DataFrame(result_df)
