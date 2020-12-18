import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import random
from datetime import datetime
import numba as nb

######################################
# Note: 
#   1. load_PreprocessData is to load all raw data and do smoothing, de-trend processing
#         and then segment series data using lists of indices.
#   2. load_dataset is to return the formatted trainable dataset in pandas dataframe format.
#       It should be run after load_PreProcessData
#   3. loadshmfile is to load .shm file directly and return numpy time series data 
#      There are 6 axis gyroscope, accelerator data in .shm file
#   4. date time information of data is in -events.txt  files, which indicates start,end, eating time period
#      We can use it for more feature extraction
###################################

## Global settings for current dataset
ACC_THRESH = 0.008   # sum(acc) (G) max value for stddev rest
GYRO_THRESH = 0.04 * 180.0/ math.pi
shimmer_global_mean = [-0.012359981,-0.0051663737,0.011612018,
                0.05796114,0.1477952,-0.034395125 ]

shimmer_global_stddev = [0.05756385,0.040893298,0.043825723,
                17.199743,15.311142,21.229317 ]

shimmer_trended_mean = [-0.000002,-0.000002,-0.000000,
                0.058144,0.147621,-0.033260 ]

shimmer_trended_stddev = [0.037592,0.034135,0.032263,
                17.209038,15.321441,21.242532 ]

all_zero_means = [0,0,0,0,0,0]

actigraph_global_means = [ 0.010016, -0.254719, 0.016803, 
                0.430628, 0.097660, 0.359574 ]

actigraph_trended_means = [ -0.000001, 0.000022, -0.000002, 
                0.430628, 0.097660, 0.359574 ]
meanvals = all_zero_means
stdvals = shimmer_trended_stddev



def loadshmfile(File_Name):
    """
    Load shm data file
    Output:
        6-axis gyroscope, accelerator time series data in numpy format
       The first 3 columns are accelerator data x, y,z
       The last 3 columns are gyroscope data x, y,z
       accelerator: col0, x: forward-backward (沿着手臂方向)  col1, y:  left-right(垂直手臂axis方向移动)，， col2, z:  up-down(上下平移), 
       gyroscope:  col 3,yaw: 水平旋转，col4, pitch 平行手臂的前后翻转，  col5, roll 绕着手臂环的axis左右翻转旋转， 
    """
    
    MB = 1024 * 1024
    RawData = np.fromfile(File_Name, dtype=np.dtype("6f4")) 
    #print(f"File {File_Name} Loaded")
    #print(sys.getsizeof(RawData)/MB)
    # Swap gyroscope axis. Remember python always uses variables with reference.
    # Swap Acceleromter
    Temp = np.copy(RawData[:,5])
    Temp2 = np.copy(RawData[:,3])
    Temp3 = np.copy(RawData[:,4])
    RawData[:,3], RawData[:,4],RawData[:,5] = Temp,Temp2, Temp3
    del Temp
    del Temp2
    del Temp3
    
    return RawData


def smooth(RawData, WINDOW_SIZE = 15,SIG = 10.0 ):
    """
    Smooth 6 axis raw data by convolution using a window
    
    """
    # Create kernel
    # size of window
    r_array = np.linspace(14,0, 15)
    Kernel = np.exp((0-np.square(r_array))/(2 * SIG * SIG))
    deno = sum(Kernel)
    Kernel = Kernel / deno
    del r_array
    del deno
    #Clone (deep copy) the variable, instead of reference. We don't want to change RawData
    Smoothed = np.copy(RawData) 
    r,c = RawData.shape
    for x in range(c):
        # Convolution followed by discarding of extra values after boundary
        Smoothed[:,x] = np.convolve(RawData[:,x], Kernel)[0:len(RawData)]
    # Copy first 15 values from Rawdata to Smoothed. np.convolve doesn't do this.
    Smoothed[:15, :] = RawData[:15,:]
    return Smoothed



def loadEvents(filename ,debug_flag=False, root_path="../data-file-indices/CAD/"):
    """
    loads events data given the .shm filename
    and parse the event.txt file to obtain meal duration
    Input: 
        filename:  <filename>-events.txt name of label file we want to load
    output:
        TotalEvents: amount of event loaded
        EventStart: a list of starting moment of meals
        EventEnd: a list of ending moment of meals
        EventNames: name of meal in string
    """
    # Load the meals file to get any triaged meals.
    SkippedMeals = []
    
    print("Loading File: ", filename)
    mealsfile = open( root_path +"meals-shimmer.txt", "r") 
    
    for line in mealsfile:
        #print(line)
        data = line.split()
        #print(data[0], data[1], data[13])
        if(int(data[13]) == 0):
            Mdata = [data[0][-9:], data[1], int(data[13])]
            SkippedMeals.append(Mdata)
    
    EventsFileName = filename.replace(".shm","-events.txt")
    
    # Load the meals
    EventNames = []
    EventStart = (np.zeros((100))).astype(int)
    EventEnd = (np.zeros((100))).astype(int)
    TotalEvents = 0
    TimeOffset = 0
    file = open(EventsFileName, "r") 
    for lines in file:
        
        words = lines.split()
        if debug_flag:
            print("Words:", words)
            
        if(len(words) == 0): continue # Skip empty lines
        # Convert Start time to offset
        if(words[0] == "START"): # Get Start Time (TimeOffset) from file
            #print(words)
            if words[2].count(":") <2:
                words[2] = words[2] +":00"
            if debug_flag:
                print(words[2])
            hours = int(words[2].split(":")[0])
            minutes = int(words[2].split(":")[1])
            seconds = int(words[2].split(":")[2])
    
            #print("{}h:{}m:{}s".format(hours, minutes,seconds))
            TimeOffset = (hours * 60 * 60) + (minutes * 60) + seconds
            continue
        if(words[0] == "END"):
            continue
         
        # word index 
        word_index = 0
        count = 0
        if debug_flag:
            print("Debug: ",words)
        while count <2 and word_index < len(words): # Process Events Data
            # skip all not- numeric string
            if words[word_index].replace(":","").isnumeric():
                count += 1
                # check if time format is correct, if not, add ":00"
                if words[word_index].count(":") <2:
                    words[word_index] = words[word_index] +":00"
                    
                if debug_flag:
                    print(words[word_index])

                hours = int(words[word_index].split(":")[0])
                minutes = int(words[word_index].split(":")[1])
                seconds = int(words[word_index].split(":")[2])
                EventTime = (hours * 60 * 60) + (minutes * 60) + seconds
                EventTime = EventTime - TimeOffset
                if(count == 1): EventStart[TotalEvents] = EventTime * 15
                if(count == 2): EventEnd[TotalEvents] = EventTime * 15
            
            word_index += 1
            
        if(TotalEvents>0):
            if(EventStart[TotalEvents]<EventStart[TotalEvents-1]):
                EventStart[TotalEvents] = EventStart[TotalEvents] + (24*60*60*15)
            if(EventEnd[TotalEvents]<EventEnd[TotalEvents-1]):
                EventEnd[TotalEvents] = EventEnd[TotalEvents] + (24*60*60*15)
        
        
        # Check if meal was triaged out for too much walking or rest
        ename = words[0]
        fname = filename[-9:]
        skipmeal = 0
        #print(fname, ename)
        for skippedmeal in SkippedMeals:
            Pname, EventName, Keep = skippedmeal
            if(Pname == fname and ename == EventName):
                #print(Pname, EventName, Keep, ename, fname, Pname == fname, ename == EventName)
                skipmeal = 1
                break
        
        if(skipmeal == 1): continue
        TotalEvents = TotalEvents + 1
        EventNames.append(ename)
    return TotalEvents, EventStart, EventEnd, EventNames, TimeOffset


def detrend(data, trend_window = 150):
    # Remove acceleration time series trend
    mean = []
    for j in range(3):
        dat = pd.Series(data[:,j]).rolling(window=trend_window).mean()
        dat[:trend_window-1] = 0
        mean.append(dat)
    mean2 = np.asarray(mean).transpose()
    data[:,0:3]-=mean2
    del mean2, mean, dat
    return data 



def load_PreprocessData(winlength, step, meanvals, stdvals,ratio_dataset=1, files_list='batch-unix.txt', 
                 removerest=1, removewalk=0, remove_trend=1, smooth_flag = 1, normalize_flag=1,shx=1, gtperc = 0.5,root_path="../data/", 
                        tqdm_flag=False, debug_flag=False):
    """
    Input:
        winlength: number of data point in a window
        step: number of data point to skip during moving the window forward
        removerest / removewalk: flag to indicate if remove rest/walk period or not
        gtperc:  ground truth percentage:  the ratio of true label in a window to the total window size
                indicating how many data points label = 1 in a window can be regarded as eating d=label
        shx：flag to indicate load shx file or not
        
        ratio_dataset: the ratio of number of days to sample over all days(354 days).
                        It controls the size of dataset. It should be in [0,1]. Otherwise return 0 sample
        
    Output:
        len(df["Filenames"]) : amount of day of dataset
        preprocessed_data: normalized, smoothed dataset
        
        samples_indices: list of indices of sample frame in dataset, 
                        format: [(i^th day, start time of window, end time of window),... ]
        labels_indices: labels corresponding to each segmentation/window
   Data Structure:
       preprocessed_data:
       [one day data: [dataframe list[axis1: [],axis2: [],axis3: [],axis4: [],axis5: [],axis6: []]....  ],
       [],
       [],
       ...
       ]
      
       samples_indices: 
       [ dataframe:[x_day, start_time, end_time] ... ] ,
       [],
       [],
       ...
       ]
       label_indices:
       [label of a dataframe  ,
       ...
       ]
       
        
    """
    ### Load data, make samples 

    samples = []
    labels = []
    preprocessed_data = []
    AllIndices = []
    totaleatingrest = 0
    totaleatingwalk = 0
    flag = 0
    # load index file of all dataset
    df = pd.read_csv(files_list, names=["Filenames"])
    
    #if ratio is 1, then do nothing
    # Otherwise, Sample some days of data based on ratio
    if ratio_dataset != 1.0:
        if ratio_dataset<1 and ratio_dataset>=0:
            # sample days without replacement
            num_days = int(ratio_dataset* len(df))
            df = df.sample(num_days,replace=False)
            # reorder index
            df.index = [i for i in range(len(df))]
        else:
            df = df.sample(0,replace=False)
        
    
    
    print("Loading Dataset ...")
    # using  tqdm package to visualize process
    data_list = range(len(df["Filenames"]))
    if tqdm_flag:
        data_list = tqdm(range(len(df["Filenames"])))
    for x in data_list:
        fileeatingrest = 0
        fileeatingwalk = 0
        filesamples = []
        filelabels = []
        File_Name = root_path + df["Filenames"][x]
        RawData = loadshmfile(File_Name)
        
        ##################################
        #Data preprocessing, Smoothing here
        #################################            
        # smoothing
        if smooth_flag:
            Smoothed = smooth(RawData)
            if debug_flag:
                print("Smoothed Data")
        else:
            Smoothed = RawData
        
        
        # remove trend of series data
        # Option:  remove acceleration bias or not using a slide window to compute mean
        if(remove_trend):
            # Remove acceleration bias
            Smoothed = detrend(Smoothed, trend_window =150)
#             TREND_WINDOW = 150
#             mean = []
#             for j in range(3):
#                 dat = pd.Series(Smoothed[:,j]).rolling(window=TREND_WINDOW).mean()
#                 dat[:TREND_WINDOW-1] = 0
#                 mean.append(dat)
#             mean2 = np.roll(np.asarray(mean).transpose(), -((TREND_WINDOW//2)-1)*3) # Shift to the left to center the values
#             # The last value in mean [-75] does not match that of phoneview, but an error in one datum is acceptable
#             # The phone view code calculates mean from -Window/2 to <Window/2 instead of including it.
#             Smoothed[:,0:3]-=mean2
#             del mean2, mean, dat

        if normalize_flag:
            # Z-Normalization after removing trend of time series     
            Normalized = normalize(Smoothed,mode ="default", meanvals=meanvals, stdvals=stdvals)
            if debug_flag:
                print("Normalized Data")
        else:
            Normalized = Smoothed
        preprocessed_data.append(Normalized)
        
        
#         Normalized = np.empty_like(Smoothed)
#         for i in range(6):
#             Normalized[:,i] = (Smoothed[:,i] - meanvals[i]) / stdvals[i]
#         # Stick this Normalized data to the Full Array
#         preprocessed_data.append(np.copy(Normalized))
        
        
        if(removerest != 0):
        # remove labels for the class of rest 
            std = []
            for j in range(6):
                dat = pd.Series(Smoothed[:,j]).rolling(window=15).std(ddof=0)
                dat[:14] = 0
                std.append(dat)
            # Above doesn't center window. Left Shift all values to the left by 7 datum (6 sensors)
            std2 = np.roll(np.asarray(std).transpose(), -7*6) 
            accstd = np.sum(std2[:,:3], axis=1)
            gyrostd = np.sum(std2[:,-3:], axis=1)
            datrest = (accstd < ACC_THRESH) & (gyrostd < GYRO_THRESH)
            # Results of rest labels
            mrest = datrest.copy()

            for i in range(8,len(datrest)-7):
                if(datrest[i]==True):
                    mrest[i-7:i+8] = True
            
            del dat, datrest, gyrostd, accstd, std2, std
        
        if(removewalk!=0):
        # remove labels for the class of walking
        # remove walk period if enabled
            minv = np.zeros((3,1))
            maxv = np.zeros((3,1))
            zerocross = np.zeros((len(Smoothed),1)).astype(int)
            for j in range(3):
                minv[j]=float('inf')
                maxv[j]= -float('inf')

            for t in range(len(Smoothed)-1):
                for j in range(3):
                    if (Smoothed[t][j+3] < minv[j]):
                        minv[j]=Smoothed[t][j+3]
                    if (Smoothed[t][j+3] > maxv[j]):
                        maxv[j]=Smoothed[t][j+3]
                    if ((Smoothed[t][j+3] < 0.0)  and  (Smoothed[t+1][j+3] > 0.0)  and  (minv[j] < -5.0)):
                        zerocross[t]+=(1<<j)
                        minv[j]=float('inf')
                        maxv[j]=-float('inf')
                    if ((Smoothed[t][j+3] > 0.0)  and  (Smoothed[t+1][j+3] < 0.0)  and  (maxv[j] > 5.0)):
                        zerocross[t]+=(1<<(j+3))
                        minv[j]=float('inf')
                        maxv[j]= -float('inf')

            zc = [0 if i==0 else 1 for i in zerocross]
            del minv, maxv, zerocross

        
        del RawData, Smoothed

        
        ###################################
        # Generating labels here
        ###################################
        # Identify things as GT
        
        TotalEvents, EventStart, EventEnd, EventNames, TimeOffset = loadEvents(File_Name, debug_flag = debug_flag)
        GT = np.zeros((len(Normalized))).astype(int)
        for i in range(TotalEvents):
            #print(EventStart[i], EventStart[i], type(EventStart[i]))
            GT[EventStart[i]: EventEnd[i]+1] = 1
        
        # Generate Sample Indices (Not sample data) and Labels
        MaxData = len(Normalized)
        for t in range(0, MaxData, step):
            # x: the x^th file data in dataset
            # t: starting time of eating
            # t+ winlength:  end time of eating
            # gtperc: Ground Truth percentage
            # Generate indices of sample
            sample_indices = [x, t, t+winlength,TimeOffset]
            #Generate label
            label = int((np.sum(GT[t:t+winlength])/winlength)>=gtperc)
            
            #Change labels if the flag of removerest or removewalk is enabled
            if(label and removerest!=0): # Only ignore if in eating
                isrest = int((np.sum(mrest[t:t+winlength])/winlength)>=0.65)
                if(isrest and removerest==1): continue; # Do not consider this sample at all. Comment this if you want to move the sample to non-eating.
                elif(isrest and removerest==2): label = 0;
                else: label = 1    
            if(label and removewalk!=0): # Only ignore if in eating
                iswalk = int((np.sum(zc[t:t+winlength])/winlength)>=0.15)
                if(iswalk and removewalk==1): continue;
                elif(iswalk and removewalk==2): label=0;
                else: label = 1

                
            if(t+winlength < MaxData): # Ignore last small window. Not ignoring results in a list rather than a numpy array.
                filesamples.append(sample_indices)
                filelabels.append(label)
                
        #merge sample indices of one day to the whole data list
        samples += filesamples
        labels += filelabels
        numsamples = len(filesamples)
        totaleatingwalk += fileeatingwalk

    samples_indices = np.asarray(samples)
    labels_array = np.asarray(labels)
    
    return len(df["Filenames"]), preprocessed_data, samples_indices, labels_array



def normalize(data,mode ="default", meanvals=all_zero_means, stdvals=shimmer_trended_stddev):
    """
    Normalize data using given mean values and std variance values
    Input:
         data: time series data set to normalize. data is one day data with 6 channels
         meanvals: global mean value of all days dataset used to smooth per day time series data
         stdvals: global standard variance.
    Output:
        smoothed data
    
    """
    
    data_normalized = np.empty_like(data)
    # Normalize
    for i in range(6):
        if mode =="default":
            data_normalized[:,i] = (data[:,i] - np.mean(data[:,i]))/ np.std(data[:,i])
        else:
            data_normalized[:,i] = (data[:,i] - np.mean(meanvals[i]))/ np.std(stdvals[i])
        
    return data_normalized




def load_dataset(data, data_indices, label_indices,raw_data_flag =True ,shuffle_flag = False, 
                 undersampling= False, enabled_time =True,test_split_ratio =0.3):
    """
    Note: need to run load_PreprocessData to get processed time series data and segmentation data
            before running this function
    Input:
        data: normalized smoothed time series data.

        sample_indices: a list of tuples containing start and end indices of each window in data
                Format:   there are n rows in data, each represents sensor data for one day
                        in each row.
                        There are m x k data, where m = amount of data point along time
                        K = amount of features in training set
        labels_array: label of each segmentation in dataset
    Output:
        pandas data frame with columns "data" and "label"
        used to train model
    """
    from tqdm import tqdm
    import sys
    outfile = sys.stdout
    
    dataset = pd.DataFrame(columns=['data','label'])
    if raw_data_flag:
        for i in tqdm(range(len(label_indices))):
            f,start_time, end_time = data_indices[i,0], data_indices[i,1], data_indices[i,2]
            sample = data[f][start_time : end_time]
            
            if enabled_time:
                time_offset = data_indices[i,3]
                freq = 1.0/15.0
                time_feat = np.array([[i for i in range(len(sample))]],dtype=float).transpose()
                time_feat *= freq
                time_feat += float(start_time)
#                 print("Sample shape: ", sample.shape, "time shape:", time_feat.shape)
                sample = np.concatenate((sample, time_feat),axis=1)
            label = label_indices[i]
            df = pd.DataFrame({'data':[sample], 'label':label})
            dataset = dataset.append(df,ignore_index=True)
    
    return dataset


def load_train_test_data(data_file_list ="../data-file-indices/CAD/"+ 'batch-unix.txt',
                         load_splitted_dataset = False,
                         ratio_dataset=1, undersampling= True,
                         test_split_ratio = 0.3,
                         winmin = 6, stridesec = 15, 
                         enabled_time_feat = True,
                         gtperc = 0.5,
                         removerest = 0, 
                         removewalk = 0, 
                         remove_trend = 0, 
                         smooth_flag = 1, normalize_flag=1,
                         data_root_path ="../data/",debug_flag=False ):
    
    # data_file_list: the name of a file containing list of shm files to load
    # data_root_path: the name of the directory storing all real data
    
#         from datetime import datetime
        winlength = int(winmin * 60 * 15)
        step = int(stridesec * 15)
        meanvals, stdvals =  all_zero_means, shimmer_trended_stddev
        
    

        files_counts, data, samples_indices, labels_array = load_PreprocessData(winlength,
                                                                                    step,
                                                                                    meanvals, 
                                                                                    stdvals,
                                                                                    
                                                                                ratio_dataset =ratio_dataset,
                                                                                    files_list=data_file_list, 
                                                                                    removerest=removerest,
                                                                                    removewalk=removewalk,
                                                                                    remove_trend=remove_trend,
                                                                                    smooth_flag = smooth_flag, 
                                                                                   normalize_flag=normalize_flag,
                                                                                    gtperc = gtperc,
                                                                                
                                                                                    debug_flag=debug_flag,
                                                                                    root_path=data_root_path)
        
        dataset = None
        if load_splitted_dataset:
            dataset = load_dataset(data, samples_indices,
                                         labels_array,undersampling= undersampling,
                                         enabled_time = enabled_time_feat,
                                         test_split_ratio=test_split_ratio)

        return dataset, data, samples_indices, labels_array
