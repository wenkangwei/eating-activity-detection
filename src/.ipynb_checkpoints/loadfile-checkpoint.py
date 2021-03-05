import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import math

ACC_THRESH = 0.008   # sum(acc) (G) max value for stddev rest
GYRO_THRESH = 0.04 * 180.0/ math.pi

def loadshmfile(File_Name):
    MB = 1024 * 1024
    RawData = np.fromfile(File_Name, dtype=np.dtype("6f4")) 
    #print(f"File {File_Name} Loaded")
    #print(sys.getsizeof(RawData)/MB)
    
    # Swap gyroscope axis. Remember python always uses variables with reference.
    # Swap Acceleromter
    Temp = np.copy(RawData[:,5])
    Temp2 = np.copy(RawData[:,3])
    Temp3 = np.copy(RawData[:,4])
    RawData[:,3] = Temp
    RawData[:,4] = Temp2
    RawData[:,5] = Temp3
    Temp =[]
    Temp2 = []
    Temp3 = []
    del Temp
    del Temp2
    del Temp3
    
    return RawData

def loadshx(File_Name):
    MB = 1024 * 1024
    RawData = np.fromfile(File_Name, dtype=np.dtype("9f4")) 
    print(f"File {File_Name} Loaded")
    print(sys.getsizeof(RawData)/MB)
    
    # Swap gyroscope axis. Remember python always uses variables with reference.
    # Swap Acceleromter
    Temp = np.copy(RawData[:,5])
    Temp2 = np.copy(RawData[:,3])
    Temp3 = np.copy(RawData[:,4])
    RawData[:,3] = Temp
    RawData[:,4] = Temp2
    RawData[:,5] = Temp3
    Temp =[]
    Temp2 = []
    Temp3 = []
    del Temp
    del Temp2
    del Temp3
    
    return RawData

def smooth(RawData):
    # Create kernel
    SIG = 10.0
    WINDOW_SIZE = 15 # size of window
    r_array = np.linspace(14,0, 15)
    Kernel = np.exp((0-np.square(r_array))/(2 * SIG * SIG))
    deno = sum(Kernel)
    Kernel = Kernel / deno
    r_array = []
    del r_array
    deno = []
    del deno
    del SIG
    del WINDOW_SIZE
    
    Smoothed = np.copy(RawData) #Clone (deep copy) the variable, instead of reference. We don't want to change RawData

    r,c = RawData.shape
    
    for x in range(c):
        Smoothed[:,x] = np.convolve(RawData[:,x], Kernel)[0:len(RawData)]# Convolution followed by discarding of extra values after boundary

    # Copy first 15 values from Rawdata to Smoothed. np.convolve doesn't do this.
    Smoothed[:15, :] = RawData[:15,:]
    return Smoothed

def loadEvents(filename):
    """
    loads events data given the .shm filename
    """
    # Load the meals file to get any triaged meals.
    SkippedMeals = []
    mealsfile = open("../data-file-indices/CAD/meals-shimmer.txt", "r") 
    for line in mealsfile:
        #print(line)
        data = line.split()
        #print(data[0], data[1], data[13])
        if(int(data[13]) == 0):
            Mdata = [data[0][-9:], data[1], int(data[13])]
            SkippedMeals.append(Mdata)
    
    EventsFileName = filename[:len(filename)-4]+"-events.txt"
    
    # Load the meals
    EventNames = []
    EventStart = (np.zeros((100))).astype(int)
    EventEnd = (np.zeros((100))).astype(int)
    TotalEvents = 0
    TimeOffset = 0
    file = open(EventsFileName, "r") 
    #print(filename)
    for lines in file:
        #print(lines)
        words = lines.split()
        if(len(words) == 0): continue # Skip empty lines
        # Convert Start time to offset
        if(words[0] == "START"): # Get Start Time (TimeOffset) from file
            #print(words)
            hours = int(words[2].split(":")[0])
            minutes = int(words[2].split(":")[1])
            seconds = int(words[2].split(":")[2])
            #print("{}h:{}m:{}s".format(hours, minutes,seconds))
            TimeOffset = (hours * 60 * 60) + (minutes * 60) + seconds
            continue
        if(words[0] == "END"):
            #print(words)
            continue
        for x in range(1,3): # Process Events Data
            hours = int(words[x].split(":")[0])
            minutes = int(words[x].split(":")[1])
            seconds = int(words[x].split(":")[2])
            EventTime = (hours * 60 * 60) + (minutes * 60) + seconds
            EventTime = EventTime - TimeOffset
            if(x == 1): EventStart[TotalEvents] = EventTime * 15
            if(x == 2): EventEnd[TotalEvents] = EventTime * 15
        if(TotalEvents>0):
            if(EventStart[TotalEvents]<EventStart[TotalEvents-1]):
                EventStart[TotalEvents] = EventStart[TotalEvents] + (24*60*60*15)
            if(EventEnd[TotalEvents]<EventEnd[TotalEvents-1]):
                EventEnd[TotalEvents] = EventEnd[TotalEvents] + (24*60*60*15)
        #print(TotalEvents)
        
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
    return TotalEvents, EventStart, EventEnd, EventNames

# Does not normalize. Normalization is done post-hoc.
def loadAllData3(winlength, step, removerest=1, removewalk=0, removebias=1, shx=1, gtperc = 0.5, file_name='batch-unix.txt'):
    ### Load data, make samples 

    samples = []
    labels = []
    AllSmoothed = []
    AllIndices = []
    totaleatingrest = 0
    totaleatingwalk = 0
    df = pd.read_csv(file_name, names=["Filenames"])
    for x in tqdm(range(len(df["Filenames"]))):
        fileeatingrest = 0
        fileeatingwalk = 0
        filesamples = []
        filelabels = []
        File_Name = "../data/" + df["Filenames"][x]
        RawData = loadshmfile(File_Name)
        Smoothed = smooth(RawData)
        Normalized = np.empty_like(Smoothed)

        if(removebias):
            # Remove acceleration bias
            TREND_WINDOW = 150
            mean = []
            for j in range(3):
                dat = pd.Series(Smoothed[:,j]).rolling(window=TREND_WINDOW).mean()
                dat[:TREND_WINDOW-1] = 0
                mean.append(dat)

            mean2 = np.roll(np.asarray(mean).transpose(), -((TREND_WINDOW//2)-1)*3) # Shift to the left to center the values
            # The last value in mean [-75] does not match that of phoneview, but an error in one datum is acceptable
            # The phone view code calculates mean from -Window/2 to <Window/2 instead of including it.

            Smoothed[:,0:3]-=mean2
            del mean2, mean, dat
        
        AllSmoothed.append(np.copy(Smoothed))
        
        if(removerest != 0):
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
            mrest = datrest.copy()

            for i in range(8,len(datrest)-7):
                if(datrest[i]==True):
                    mrest[i-7:i+8] = True
            
            del dat, datrest, gyrostd, accstd, std2, std
        
        if(removewalk!=0):
            minv = np.zeros((3,1))
            maxv = np.zeros((3,1))
            zerocross = np.zeros((len(Smoothed),1)).astype(int)
            for j in range(3):
                minv[j]=999.9
                maxv[j]=-999.9

            for t in range(len(Smoothed)-1):
                for j in range(3):
                    if (Smoothed[t][j+3] < minv[j]):
                        minv[j]=Smoothed[t][j+3]
                    if (Smoothed[t][j+3] > maxv[j]):
                        maxv[j]=Smoothed[t][j+3]
                    if ((Smoothed[t][j+3] < 0.0)  and  (Smoothed[t+1][j+3] > 0.0)  and  (minv[j] < -5.0)):
                        zerocross[t]+=(1<<j)
                        minv[j]=999.9
                        maxv[j]=-999.9
                    if ((Smoothed[t][j+3] > 0.0)  and  (Smoothed[t+1][j+3] < 0.0)  and  (maxv[j] > 5.0)):
                        zerocross[t]+=(1<<(j+3))
                        minv[j]=999.9
                        maxv[j]=-999.9

            zc = [0 if i==0 else 1 for i in zerocross]
            del minv, maxv, zerocross
        
        del RawData

        # Identify things as GT
        [TotalEvents, EventStart, EventEnd, EventNames] = loadEvents(File_Name) #loadfile.loadEvents(File_Name)
        GT = np.zeros((len(Smoothed))).astype(int)
        for i in range(TotalEvents):
            #print(EventStart[i], EventStart[i], type(EventStart[i]))
            GT[EventStart[i]: EventEnd[i]+1] = 1

        # Generate labels 
        MaxData = len(Smoothed)
        for t in range(0, MaxData, step):
            sample = [x, t, t+winlength]
            label = int((np.sum(GT[t:t+winlength])/winlength)>=gtperc)
            #isrest =  if(removerest) else 0
            #iswalk =  if(removewalk) else 0
            #if(label and isrest):
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
#                fileeatingwalk+=1
#                continue # Do not append this sample to the dataset
                
            if(t+winlength < MaxData): # Ignore last small window. Not ignoring results in a list rather than a numpy array.
                filesamples.append(sample)
                filelabels.append(label)

        samples = samples + filesamples
        labels = labels + filelabels
        numsamples = (len(filesamples))
        totaleatingwalk+=fileeatingwalk
        #print("Loaded file {}, {} samples from {}".format(x, numsamples,File_Name), flush=True)
        #print("Loaded file {}, {} samples from {}, contains {} rest in eating".format(x,numsamples,File_Name,fileeatingrest),
        #      flush=True)

    samples_array = np.asarray(samples)
    labels_array = np.asarray(labels)
    #print("Total {:d} walking in eating\n".format(fileeatingwalk))
    return len(df["Filenames"]), AllSmoothed, samples_array, labels_array

# Reads from designated file. Useful for handedness and grouping. Does not normalize. Normalization is done post-hoc.
def loadAllData4(filename, winlength, step, removerest=1, removewalk=0, removebias=1, shx=1, gtperc = 0.5):
    ### Load data, make samples 

    samples = []
    labels = []
    AllSmoothed = []
    AllIndices = []
    totaleatingrest = 0
    totaleatingwalk = 0
    df = pd.read_csv(filename, names=["Filenames"])
    for x in tqdm(range(len(df["Filenames"]))):
    #for x in tqdm(range(10)):
        fileeatingrest = 0
        fileeatingwalk = 0
        filesamples = []
        filelabels = []
        File_Name = "/home/spsharm/" + df["Filenames"][x]
        RawData = loadshmfile(File_Name) #loadfile.loadshmfile(File_Name)
        Smoothed = smooth(RawData) #loadfile.smooth(RawData)    
        Normalized = np.empty_like(Smoothed)

        if(removebias):
            # Remove acceleration bias
            TREND_WINDOW = 150
            mean = []
            for j in range(3):
                dat = pd.Series(Smoothed[:,j]).rolling(window=TREND_WINDOW).mean()
                dat[:TREND_WINDOW-1] = 0
                mean.append(dat)

            mean2 = np.roll(np.asarray(mean).transpose(), -((TREND_WINDOW//2)-1)*3) # Shift to the left to center the values
            # The last value in mean [-75] does not match that of phoneview, but an error in one datum is acceptable
            # The phone view code calculates mean from -Window/2 to <Window/2 instead of including it.

            Smoothed[:,0:3]-=mean2
            del mean2, mean, dat
        
        AllSmoothed.append(np.copy(Smoothed))
        
        if(removerest != 0):
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
            mrest = datrest.copy()

            for i in range(8,len(datrest)-7):
                if(datrest[i]==True):
                    mrest[i-7:i+8] = True
            
            del dat, datrest, gyrostd, accstd, std2, std
        
        if(removewalk!=0):
            minv = np.zeros((3,1))
            maxv = np.zeros((3,1))
            zerocross = np.zeros((len(Smoothed),1)).astype(int)
            for j in range(3):
                minv[j]=999.9
                maxv[j]=-999.9

            for t in range(len(Smoothed)-1):
                for j in range(3):
                    if (Smoothed[t][j+3] < minv[j]):
                        minv[j]=Smoothed[t][j+3]
                    if (Smoothed[t][j+3] > maxv[j]):
                        maxv[j]=Smoothed[t][j+3]
                    if ((Smoothed[t][j+3] < 0.0)  and  (Smoothed[t+1][j+3] > 0.0)  and  (minv[j] < -5.0)):
                        zerocross[t]+=(1<<j)
                        minv[j]=999.9
                        maxv[j]=-999.9
                    if ((Smoothed[t][j+3] > 0.0)  and  (Smoothed[t+1][j+3] < 0.0)  and  (maxv[j] > 5.0)):
                        zerocross[t]+=(1<<(j+3))
                        minv[j]=999.9
                        maxv[j]=-999.9

            zc = [0 if i==0 else 1 for i in zerocross]
            del minv, maxv, zerocross
        
        del RawData

        # Identify things as GT
        [TotalEvents, EventStart, EventEnd, EventNames] = loadEvents(File_Name) #loadfile.loadEvents(File_Name)
        GT = np.zeros((len(Smoothed))).astype(int)
        for i in range(TotalEvents):
            #print(EventStart[i], EventStart[i], type(EventStart[i]))
            GT[EventStart[i]: EventEnd[i]+1] = 1

        # Generate labels 
        MaxData = len(Smoothed)
        for t in range(0, MaxData, step):
            sample = [x, t, t+winlength]
            label = int((np.sum(GT[t:t+winlength])/winlength)>=gtperc)
            #isrest =  if(removerest) else 0
            #iswalk =  if(removewalk) else 0
            #if(label and isrest):
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
#                fileeatingwalk+=1
#                continue # Do not append this sample to the dataset
                
            if(t+winlength < MaxData): # Ignore last small window. Not ignoring results in a list rather than a numpy array.
                filesamples.append(sample)
                filelabels.append(label)

        samples = samples + filesamples
        labels = labels + filelabels
        numsamples = (len(filesamples))
        totaleatingwalk+=fileeatingwalk
        #print("Loaded file {}, {} samples from {}".format(x, numsamples,File_Name), flush=True)
        #print("Loaded file {}, {} samples from {}, contains {} rest in eating".format(x,numsamples,File_Name,fileeatingrest),
        #      flush=True)

    samples_array = np.asarray(samples)
    labels_array = np.asarray(labels)
    #print("Total {:d} walking in eating\n".format(fileeatingwalk))
    return len(df["Filenames"]), AllSmoothed, samples_array, labels_array

# Global Dataset Normalization with wider range (smaller SD)
def globalZscoreNormalize2(AllSmoothed, meanvals, stdvals, denomscalingfactor):
    
    AllNormalized = []
    
    for x in range(len(AllSmoothed)):
        Smoothed = AllSmoothed[x]
        Normalized = np.empty_like(Smoothed)
        # Normalize
        for i in range(6):
            Normalized[:,i] = (Smoothed[:,i] - meanvals[i]) / (stdvals[i] * denomscalingfactor)
        
        # Stick this Normalized data to the Full Array
        AllNormalized.append(np.copy(Normalized))
    
    return AllNormalized