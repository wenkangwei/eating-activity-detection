import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def saveHistoryPlot(H, foldername, filename):
    fig = plt.figure()
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('Model mean squared error loss')
    plt.ylabel('Mean squared error loss')
    plt.xlabel('Epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    fig.savefig(foldername+"loss_"+filename, dpi=fig.dpi)
    
    fig2 = plt.figure()
    plt.plot(H.history['acc'])
    plt.plot(H.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    fig2.savefig(foldername+"acc_"+filename, dpi=fig.dpi)

def getEatingMetrics(GTlabels, predicted_labels):
    TN, FP, FN, TP = confusion_matrix(GTlabels, predicted_labels).ravel()
    E = TP + FN
    NE = TN + FP
    PP = TP + FP
    Precision = TP / (PP)
    Recall = TP / (E) # Sensitivity for Eating
    Sensitivity = Recall
    Specificity = TN / NE # Recall for NE
    Wacc = (20*TP + TN)/(20*E + NE)
    return TN, FP, FN, TP, Sensitivity, Specificity, Wacc
    #print("Recall (Sensitivity): {:.2f}, NE Recall (Specificity): {:.2f}, WAcc: {:.2f}".format(Recall, Specificity, Wacc))
    
######################################
### Save History after every epoch ###
######################################

import json, codecs
import os
def saveHist(path, history):
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4) 

def loadHist(path):
    n = {} # set history to empty
    if os.path.exists(path): # reload history if it exists
        with codecs.open(path, 'r', encoding='utf-8') as f:
            n = json.loads(f.read())
    return n

def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest

from keras.callbacks import Callback
class LossHistory(Callback):
    
    def __init__(self, hfname):
        self.hfname = hfname
        
    def on_train_begin(self, logs={}):
        self.current_history = loadHist(self.hfname)
    
    # https://stackoverflow.com/a/53653154/852795
    def on_epoch_end(self, epoch, logs = None):
        new_history = {}
        for k, v in logs.items(): # compile new history from logs
            new_history[k] = [v] # convert values into lists
        #current_history =  # load history from current training Avoid reloading every time
        self.current_history = appendHist(self.current_history, new_history) # append the logs
        saveHist(self.hfname, self.current_history) # save history from current training