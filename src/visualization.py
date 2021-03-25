from __future__ import print_function
from packages import *

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


def plot_threshold_results(threshold_results, te_val = 0.1, ts_val=0.6,figsize=(12,10),legend_out=True, grid =False):
    """
    plot threshold results generated from test_threshold_csv()  function in cross_validation.py file
    """
    print("Fixed Te, Change Ts")
    fig, ax= plt.subplots(2,2, figsize=figsize)
    
    for j in range(2):
            ax[0,j].set_xlim(threshold_results['Ts'].min(),threshold_results['Ts'].max())
    
    for j in range(2):
            ax[1,j].set_xlim(threshold_results['Te'].min(),threshold_results['Te'].max())
            
    th_df = threshold_results[(threshold_results['Te']==te_val)]
    
    fig_ts1 = sns.lineplot(data=th_df, x="Ts", y= "TPR", hue="dataset",ax=ax[0,0])
    fig_ts2 = sns.lineplot(data=th_df, x="Ts", y= "FP/TP", hue="dataset",ax=ax[0,1])
    _ = ax[0,0].set_title("Te = "+str(te_val))
    _ = ax[0,1].set_title("Te = "+str(te_val))

    
    th_df = threshold_results[(threshold_results['Ts']==ts_val)]
    fig_te1 = sns.lineplot(data=th_df, x="Te", y= "TPR", hue="dataset",ax=ax[1,0])
    fig_te2 = sns.lineplot(data=th_df, x="Te", y= "FP/TP", hue="dataset",ax=ax[1,1])
    _ = ax[1,0].set_title("Ts = "+str(ts_val))
    _ = ax[1,1].set_title("Ts = "+str(ts_val))
    if grid:
        fig_ts1.grid()
        fig_ts2.grid()
        fig_te1.grid()
        fig_te2.grid()
    if not legend_out :
        fig_ts1.legend(loc="upper right")
        fig_ts2.legend(loc="upper right")
        fig_te1.legend(loc="upper right")
        fig_te2.legend(loc="upper right")
    else:
        fig_te2.legend_.remove()
        fig_ts1.legend_.remove()
        fig_te1.legend_.remove()
        fig_ts2.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    
    return [fig_ts1,fig_ts2], [fig_te1,fig_te2]

def save_figs(fig_ts, fig_te, dpi=400,fig_path = "../results/images/",prefix=""):
    
    fig = fig_ts[0].get_figure()
    fig.savefig(fig_path+prefix+"fig_ts1", dpi = dpi)
    fig = fig_ts[1].get_figure()
    fig.savefig(fig_path+prefix+"fig_ts2", dpi = dpi)
    fig = fig_te[0].get_figure()
    fig.savefig(fig_path+prefix+"fig_te1", dpi = dpi)
    fig = fig_te[1].get_figure()
    fig.savefig(fig_path+prefix+"fig_te2", dpi = dpi)


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
        threshold = {}
        for name in names:
            threshold[name] = [0.8,0.4]
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


def generate_possibility(dataset,fold_num=5):
    """
    dataset: person_meal dataset
    output: generated possibility adn ground true label and prediction in episode
    """
    result = pd.DataFrame()
    from sklearn.model_selection import KFold
    days = np.unique(dataset.data_indices[:,0])
    kf = KFold(n_splits=5, random_state= 1000,shuffle=False)
    
    for fold, (day_train_idx, day_test_idx) in enumerate(kf.split(days)):
        day_test_idx = day_test_idx.tolist()
        proba_path ="../results/possibility_results/{}/cv_fold_{}_".format(dataset.person_name,fold)
        partial_result = hysteresis_threshold(None, dataset,days_ls = day_test_idx,start_threshold=0.8, end_threshold=0.4,
                                                  winmin = 6, stepsec=5, episode_min = 1.,
                                                 load_proba_flag=True, path =proba_path)
        #print(partial_result)
        result = result.append(partial_result,ignore_index=True)
    proba_ls, labels_ls, preds_ls =map_results(dataset,result)
    return proba_ls, labels_ls, preds_ls, result




def visualize_prob(offset,winsize,day, model_result="I",file_name ="possibility_seq" ):
    """
    Note:
        proba_ls, labels_ls, preds_ls are global variables from notebook
    """
    stride  = 5 *15 # 5 seconds between two adjacent labels/window samples
    fig_path = "../results/images/"
    day = int(day)
    global proba_ls
    global labels_ls
    global preds_ls
    if model_result =="I":
        proba_list=proba_ls
        labels_list= labels_ls
        preds_list=preds_ls
    else:
        proba_list=proba_ls_g
        labels_list= labels_ls_g
        preds_list=preds_ls_g
    proba, labels, preds = proba_list[day], labels_list[day], preds_list[day]    
    
    offset = offset *15
    if winsize == -1:
        winsize = len(labels)
        offset = 0
    else:
        winsize = winsize*15
    
    if  len(labels)-winsize <0:
        #offset = len(labels)-winsize
        offset =0
        winsize = len(labels)
        
    t = np.arange(start = offset, stop= offset+winsize, step=1)
    print("Offset: ",offset, "winszie: ",winsize,"t shape: ",t.shape, "label shape:", labels.shape, preds.shape)
    fig, ax = plt.subplots(3,1,figsize= (20,12))
    df1 = proba[offset:offset+winsize]
    df2 = np.array(preds[offset:offset+winsize]) #*10-5
    df3 = np.array(labels[offset:offset+winsize])#*10-5
    x1= sns.lineplot(x=t, y=df1 , ax =ax[0],color= 'grey',label="Possibility")
    x2 = sns.lineplot(t,df2 , ax =ax[1],color='g', linewidth=1.5,label="Prediction(Eat)")
    x3 = sns.lineplot(t,df3 , ax =ax[2],color='b', linewidth=1.5, label="Label(Eat)")
        
    
        
    ax[0].fill_between( t, df1, 
                interpolate=True, color='grey')
    
    ax[0].fill_between(t, df3, where=(df3==1), 
                interpolate=True, color='blue')
    ax[0].fill_between( t,df2, where=(df2==1), 
                interpolate=True, color='green')
    ax[0].set_ylim(0,1)
    print(offset,len(labels)-winsize )
    if offset >= len(labels)-winsize:
        title_txt ="Day: "+ str(day) + " Whole Day samples: "+str(len(labels)) +" . " + "Sample plotted: "+str(winsize)+". "
    else:
        title_txt = "Day: "+ str(day) + "Samples from "+str(offset) +"~" + str((offset+winsize)) +". "+ "Number of Sample plotted: "+str(winsize)+". "
    ax[0].set_title(title_txt)
    ax[2].set_xlabel("index of sample")
    ax[0].set_ylabel("Possibility")
    ax[1].set_ylabel("Predictions")
    ax[2].set_ylabel("Ground Truch")
    
    ax[0].legend(["Possibility"],loc='upper left')
    ax[1].legend(["1: Eat, 0:Other"],loc='upper left')
    ax[2].legend(["1: Eat, 0:Other"],loc='upper left')
    x3 = x3.get_figure()
    x3.savefig(fig_path+file_name, dpi = 80)
    return 
