

from data_loader import *
import numpy as np
import pandas as pd
import torch
import sklearn
import os
import random


def create_train_test_file_list(file_name= "all_files_list.txt",person_name = 'wenkanw',
                     out_path = "../data-file-indices/",root_path= "../",
                     test_ratio = 0.2, print_flag = True):
    """
    This function is used to split test set and training set based on file names
    
    """
    shm_file_ls = []
    event_file_ls = []
    new_files = []
    if person_name == "CAD":
        fp = open(out_path+ person_name+ "/" +"batch-unix.txt", "r")
        txt = fp.read()
        fp.close()
        # save all file list
        fp = open(out_path+ person_name+ "/" +"all_files_list.txt", "w")
        fp.write(txt)
        fp.close()
        
        txt_ls = txt.split("\n")
        txt_ls.remove("")
        txt_ls= [txt+"\n" for txt in txt_ls]
        test_size = int(len(txt_ls)*test_ratio)
        test = "".join(txt_ls[len(txt_ls) - test_size: ])
        train = "".join(txt_ls[:len(txt_ls) - test_size ])
        
        fp = open(out_path+ person_name+ "/" +"test_files.txt", "w")
        fp.write(test)
        fp.close()
        
        fp = open(out_path+ person_name+ "/" +"train_files.txt", "w")
        fp.write(train)
        fp.close()
        
        if print_flag:
            print("Train:", len(txt_ls) - test_size)
            print(train)
            print("test: ",test_size)
            print(test)
        return 
        
        
    for dirname, _, filenames in os.walk(root_path + 'data/IndividualData'):
        for filename in filenames:
            # check every file name in the individual data folder
            path = os.path.join(dirname, filename)
#             print("Path: ",path)
            # check if datafile is shm file and is not a test file
            if ".shm" in filename and person_name in path and 'test' not in path:
                # If the data file has label file as well, then it is valid
                # and we add it to the filename list
                event_file_name =  filename.replace(".shm","-events.txt")
                
                if event_file_name in filenames:
                    # if both shm and event files exist
                    new_file = path.replace(root_path+"data/","")
                    new_file += "\n"
                    new_files.append(new_file)
        
    new_files.sort()
    if test_ratio > 0.:
        # split train files and test files
        test_size = int(len(new_files)*test_ratio)
        test_files = new_files[:test_size]
        train_files = new_files[test_size:]
        # write train files
        fp = open(out_path+ person_name+ "/" +"train_files.txt", "w")
        train = "".join(train_files)
        
        fp.write(train)
        fp.close()
        # write test files
        fp = open(out_path+ person_name+ "/" +"test_files.txt", "w")
        test = "".join(test_files)
        fp.write(test)
        fp.close()
        
        if print_flag:
            print("Train:")
            print(train)
            print("test: ")
            print(test)
    
    fp = open(out_path+person_name+ "/"+file_name, "w")
    all_files = "".join(new_files)
    fp.write(all_files)
    fp.close()
    
    if print_flag:
        print("All files: ")
        print(all_files)
        





class Person_MealsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset = None,person_name= "wenkanw", 
                 data_indices_file = "../data-file-indices/",
                 file_name = "all_files_list",
                 remove_trend = 0,
                 remove_walk = 0,
                 remove_rest = 0,
                 smooth_flag = 1,
                 normalize_flag = 1,
                 winmin = 6,
                 stridesec = 15,
                 gtperc = 0.5,
                 device = 'cpu',
                 ratio_dataset=1,
                load_splitted_dataset = False,
                 enable_time_feat = False,
                 debug_flag= False
                ):
        
        if file_name == "train":
            file_name = data_indices_file + person_name +"/"+"train_files.txt"
        elif file_name == "test":
            file_name = data_indices_file + person_name +"/"+"test_files.txt"
        else:
            file_name = data_indices_file + person_name +"/"+ file_name+".txt"
            
        # Note: file_name is the name of file that contain the list of shm files' names
        self.file_name = file_name
        self.dataset = dataset
        
        self.winmin = winmin
        self.stridesec = stridesec
        self.load_splitted_dataset = load_splitted_dataset
        self.remove_trend = remove_trend
        self.remove_walk = remove_walk
        self.remove_rest = remove_rest
        self.smooth_flag = smooth_flag
        self.normalize_flag = normalize_flag
        self.gtperc = gtperc,
        self.ratio_dataset = ratio_dataset
        self.enable_time_feat = enable_time_feat
        self.device = device
        self.debug_flag= debug_flag
        if not self.dataset:
            self.get_data(person_name)

    def get_data(self, person_name):
            
            
            # files_counts, data, samples_indices, labels_array
            # Note: the data preprocessing in this function is for global time series dataset
            
            self.dataset, self.data, self.data_indices, self.labels = load_train_test_data(data_file_list =self.file_name,
                                    load_splitted_dataset = False,
                                     ratio_dataset=self.ratio_dataset,
                                     enabled_time_feat = self.enable_time_feat, 
                                     winmin = self.winmin, stridesec = self.stridesec,gtperc = self.gtperc,
                                     removerest = self.remove_rest,
                                     removewalk = self.remove_walk, smooth_flag = self.smooth_flag, normalize_flag=self.normalize_flag, 
                                     remove_trend = self.remove_trend,
                                     debug_flag=self.debug_flag )
            
            if self.load_splitted_dataset:
                self.dataset = self.get_dataset()
                
            
        
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        data = self.get_item(index)
        return data['data'],data['label']
        
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return  len(self.dataset) if self.load_splitted_dataset else len(self.data_indices)
    def get_item(self, index, tensor_type=True):
        """
        This function is used to obtain one sample data point
        """
        f,start_time, end_time = self.data_indices[index,0], self.data_indices[index,1], self.data_indices[index,2]
        sample = self.data[f][start_time : end_time]
        data = pd.DataFrame(columns=['data','label'])    
        # Add time feature to data
        if self.enable_time_feat:
            time_offset = self.data_indices[index,3]
            freq = 1.0/15.0
            time_feat = np.array([[i for i in range(len(sample))]],dtype=float).transpose()
            time_feat *= freq
            time_feat += float(start_time)* freq
            time_feat += time_offset
            sample = np.concatenate((sample, time_feat),axis=1)
        label = self.labels[index]
        if tensor_type:
            data = {"data":torch.tensor(sample, dtype =torch.float, device =  self.device ), 'label': label}
        else:
            data = {"data":sample, 'label': label}
        return data
    
    def get_dataset(self, start_index = None, end_index = None):
        """
        This function is used to obtain the whole dataset in pandas or part of whole dataset
        It is good to use this to sample some data to analyze
        """
        start_i = 0 if not start_index else start_index
        end_i = self.__len__() if not end_index else end_index
        
        dataset = pd.DataFrame(columns=['data','label'])
        for i in tqdm(range(start_i, end_i)):
            data = self.get_item(i)
            dataset = dataset.append(data,ignore_index=True)
        self.dataset = dataset
        return self.dataset
    
    def sample(self, num = 1000,random_state = None):
        """
        Simply sample part of data for analysis
        """
        if random_state != None:
            np.random.seed(random_state)
            
        sample_data = pd.DataFrame(columns=['data','label'])
        indices = np.random.choice(len(self.labels), num)
        for i in tqdm(indices):
            data = self.get_item(i)
            data["data"] = data["data"].numpy()
            sample_data = sample_data.append(data,ignore_index=True)
        return sample_data
    
    def get_subset(self, indices_ls):
        axdata = []
        aydata = []
        for i in indices_ls:
            data = self.get_item(i, tensor_type=False)
            sample = data['data']
            label = data['label']
            axdata.append(sample)
            aydata.append(label)
        subsetData = np.array(axdata, copy=True) # Undersampled Balanced Training Set
        subsetLabels = np.array(aydata, copy=True)
        del axdata
        del aydata
        return subsetData, subsetLabels
        
        
            
                
def balance_data_indices(labels, sample_num = 4000,mode= "under", replace = False,shuffle=True, random_state = 1000):
    """
    sample_num: number of samples of each class after balancing
    mode: 
        under - undersampling
        over - oversampling
        mix - undersampling negative samples + oversampling positive samples, each class has sample_num amount samples in this mode
    return:
        balanced indices
    """
    eat_labels_index = [i for i, e in enumerate(labels) if e >= 0.5]
    not_eat_labels_index = [i for i, e in enumerate(labels) if e < 0.5]
    eat_index = eat_labels_index
    not_eat_index = not_eat_labels_index
    if random_state != None:
        np.random.seed(random_state)
        
    if mode == "over":
        eat_index = np.random.choice(eat_labels_index,len(not_eat_labels_index)).tolist()
        pass
    elif mode == "under":
        not_eat_index = np.random.choice(not_eat_labels_index,len(eat_labels_index)).tolist()
        pass
    else:
        #default as mix
        eat_index = np.random.choice(eat_labels_index,sample_num, replace = replace).tolist()
        not_eat_index = np.random.choice(not_eat_labels_index,sample_num, replace = replace).tolist()
        pass
    
    indices_balanced = eat_index + not_eat_index
    if shuffle:
        np.random.shuffle(indices_balanced)
    
    return indices_balanced
