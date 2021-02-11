
from packages import *

def plot_threshold_results(threshold_results, te_val = 0.1, ts_val=0.6):
    """
    plot threshold results generated from test_threshold_csv()  function in cross_validation.py file
    """
    print("Fixed Te, Change Ts")
    fig, ax= plt.subplots(2,2, figsize=(12,10))

    th_df = threshold_results[(threshold_results['Te']==te_val)]
    _ = sns.lineplot(data=th_df, x="Ts", y= "TPR", hue="dataset",ax=ax[0,0])
    _ = sns.lineplot(data=th_df, x="Ts", y= "FP/TP", hue="dataset",ax=ax[0,1])
    _ = ax[0,0].set_title("Te = "+str(te_val))
    _ = ax[0,1].set_title("Te = "+str(te_val))


    th_df = threshold_results[(threshold_results['Ts']==ts_val)]
    _ = sns.lineplot(data=th_df, x="Te", y= "TPR", hue="dataset",ax=ax[1,0])
    _ = sns.lineplot(data=th_df, x="Te", y= "FP/TP", hue="dataset",ax=ax[1,1])
    _ = ax[1,0].set_title("Ts = "+str(ts_val))
    _ = ax[1,1].set_title("Ts = "+str(ts_val))
    


def map_prediction_gt(meal_data, day,possib_result ):
    """
    Convert segmentation back to binary labels
    and Map the prediction possibility sequence  back to the same shape
    """
    day = int(day)
    res = possib_result 
    possib = np.array(res.proba.iloc[day])
    step= res["stepsec"].iloc[0] *15
    start_ls, end_ls = meal_data.get_GT_segment()
    start_ls = start_ls[day]
    end_ls = end_ls[day]
    proba = np.zeros([len(meal_data.data[day]), ] )
    labels = np.zeros([len(meal_data.data[day]), ] )
    preds = np.zeros([len(meal_data.data[day]), ] )
    
    #probability sequence
    for i in range(len(possib)):
        proba[i*step: (i+1)*step]= possib[i]
    
    # GT label
    for i in range(len(start_ls)):
        labels[start_ls[i]:end_ls[i]+1] = 1
        
    # prediction label by hysteresis threshold
    seg_start_ls = res["segment_start"].iloc[day]
    seg_end_ls = res["segment_end"].iloc[day]
    for i in range(len(seg_start_ls)):
        s = int(seg_start_ls[i] * step)
        e = int(seg_end_ls[i] * step)
        preds[s:e] =1
    return proba, labels, preds

def map_results(meal_data,possib_result):
    """
    Convert segmentation back to binary labels
    and Map the prediction possibility sequence  back to the same shape
    for all days of data
    """
    proba_ls, labels_ls, preds_ls = [],[],[]

    for day in range(len(possib_result)):
        proba, labels, preds = map_prediction_gt(meal_data, day,possib_result )
        proba_ls.append(proba)
        labels_ls.append(labels)
        preds_ls.append(preds)
    return  proba_ls, labels_ls, preds_ls

def get_episode_output(names= [], threshold= None, use_group_model= False,load_proba_flag=True):
    """
    Generate probability sequences for all days of data in all dataset
    """
    output_df = {"dataset":[],"proba_ls":[],"labels_ls":[],"preds_ls":[]}
    if not threshold:
        threshold = {'wenkanw':[0.8, 0.3], 'adam':[0.8,0.3],'lawler':[0.8,0.3], 'shaurya':[0.8,0.3]}
    for person in names:
        output_df["dataset"].append(person)
        meal_data = Person_MealsDataset(person_name= person, file_name = "all_files_list", winmin = 6,stridesec = 5,smooth_flag = 1,
                         normalize_flag = 1)
        high_th, low_th = threshold[person][0], threshold[person][1]
        if use_group_model:
            model = tf.keras.models.load_model('../models/CAD_models/acti_6min_M_F_6.000000Min.h5')
            result = hysteresis_threshold(model, meal_data,start_threshold=high_th, end_threshold=low_th, winmin = 6, stepsec=5, episode_min = 1.,
                                     load_proba_flag=load_proba_flag,path="../results/possibility_results/group_")
        else:
            model = tf.keras.models.load_model('../models/'+ person + '_models/acti_6min_split_day_M_F_6.000000Min.h5')
            result = hysteresis_threshold(model, meal_data,start_threshold=high_th, end_threshold=low_th, winmin = 6, stepsec=5, episode_min = 1.,
                                     load_proba_flag=load_proba_flag)
            
        proba_ls, labels_ls, preds_ls =map_results(meal_data,result)
        output_df["proba_ls"].append(proba_ls)
        output_df["labels_ls"].append(labels_ls)
        output_df["preds_ls"].append(preds_ls)
    return pd.DataFrame(output_df)
