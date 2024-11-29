#this file contains all the preprocessing functions used in preprocessing.py
#this file is tested in test_preproessing.py

import os
import shutil
import pandas as pd
from keras.preprocessing import sequence
from sklearn import utils
import numpy as np
import matplotlib.pyplot as plt

def populate_lists():
    #takes in nothing --> returns two populated x and y lists + names file for cross reference
    all_x = []
    all_y = [] #"0:nostutter, 1: stutter"

    path1 = os.path.join("StutterData", "Stutter2")
    directory1 = os.fsencode(path1)

    path2 = os.path.join("StutterData", "NoStutter2")
    directory2 = os.fsencode(path2)

    names = []

    for file in os.listdir(directory1):
            filename = os.fsdecode(file)
            try:
                dataset = pd.read_csv(os.path.join(path1, filename), header=0, usecols=["MsBetweenPresents"], dtype = {"MsBetweenPresents":float})
                data_list = dataset["MsBetweenPresents"].tolist()

            except:
                dataset = pd.read_csv(os.path.join(path1, filename), header=0, usecols=["msBetweenPresents"], dtype = {"msBetweenPresents":float})
                data_list = dataset["msBetweenPresents"].tolist()

            #padding the sequence to 15000. if more, truncate at the end. if less, add 0's to the end
            all_x.append(data_list)
            all_y.append(1)
            names.append(filename)
    
    for file in os.listdir(directory2):
            filename = os.fsdecode(file)
            try:
                dataset = pd.read_csv(os.path.join(path2, filename), header=0, usecols=["MsBetweenPresents"], dtype = {"MsBetweenPresents":float})
                data_list = dataset["MsBetweenPresents"].tolist()

            except:
                try:
                    dataset = pd.read_csv(os.path.join(path2, filename), header=0, usecols=["msBetweenPresents"], dtype = {"msBetweenPresents":float})
                    data_list = dataset["msBetweenPresents"].tolist()
                except:
                    print("uh oh, help!")
                    
            #padding the sequence to 30000. if more, truncate at the end. if less, add 0's to the end
            all_x.append(data_list)
            all_y.append(0)
            names.append(filename)

    return all_x, all_y, names


def padding_x(all_x_before, length_seq):
    #takes in features for x and pads them to a specific length
    all_x = sequence.pad_sequences(all_x_before, maxlen=length_seq, padding="post", truncating="post", dtype= "float32")
    return all_x


def alter_spike(all_x_before, abs, factor):
    #takes in x features; is able to amplify or decrease spikes
    new_x = []
    
    for example in all_x_before:
        for ms_index in range(len(example)):
            num = example[ms_index]
            if num >= abs:
                example[ms_index] = num * factor

        new_x.append(example)

    return new_x

def shuffle(list1, list2, list3=None):
    if list3==None:
        a,b = utils.shuffle(list1, list2, random_state=115)
        return a,b
    else:
        a,b,c = utils.shuffle(list1, list2, list3, random_state=115)
        return a,b,c


def normalize_each(x_list):
    #takes in a feature and returns the normalized version; normalizes per case in the list of lists
    from sklearn.preprocessing import MinMaxScaler
    normalized = MinMaxScaler()
    x_list_2 = []
    for series in x_list:
        series = np.reshape(series,(-1, 1))
        data_std = normalized.fit_transform(series)
        # data_std = data_std[0]
        data_std = [item for sublist in data_std for item in sublist]
        x_list_2.append(data_std) 

    return x_list_2

def normalize_all(x_list, length, scaler = None):
    #takes in a feature and returns the normalized version; normalizes the entire dataset at once
    from sklearn.preprocessing import MinMaxScaler
    
    length_original = len(x_list)
    x_list_2 = []
    if scaler==None:
        normalized = MinMaxScaler()
        large_list = []
        for series in x_list:
            for num in series:
                large_list.append(num)

        large_list = np.reshape(large_list, (-1, 1))
        data_std = normalized.fit_transform(large_list)

        for time in range(length_original):
            x_list_2.append(data_std[0:length])
            data_std = data_std[length:]

        return x_list_2, normalized

    else:
        large_list = []
        for series in x_list:
            for num in series:
                large_list.append(num)
        
        large_list = np.reshape(large_list, (-1, 1))
        data_std = scaler.transform(large_list)

        for time in range(length_original):
            x_list_2.append(data_std[0:length])
            data_std = data_std[length:]

        return x_list_2

def standardize_each(x_list):
    #takes in a feature and returns the standardized version
    from sklearn.preprocessing import StandardScaler
    standardized = StandardScaler()
    x_list_2 = []
    for series in x_list:
        series = np.reshape(series,(-1, 1))
        data_std = standardized.fit_transform(series)
        # data_std = data_std[0]
        data_std = [item for sublist in data_std for item in sublist]
        x_list_2.append(data_std) 

    return x_list_2

def standardize_all(x_list, length, scaler=None):
   #takes in a feature and returns the standardized version
    from sklearn.preprocessing import StandardScaler
    length_original = len(x_list)
    x_list_2 = []
    if scaler==None:
        standardized = StandardScaler()
        large_list = []
        for series in x_list:
            for num in series:
                large_list.append(num)

        large_list = np.reshape(large_list, (-1, 1))
        data_std = standardized.fit_transform(large_list)

        for time in range(length_original):
            x_list_2.append(data_std[0:length])
            data_std = data_std[length:]

        return x_list_2, standardized

    else:
        large_list = []
        for series in x_list:
            for num in series:
                large_list.append(num)

        large_list = np.reshape(large_list, (-1, 1))
        data_std = scaler.transform(large_list)

        for time in range(length_original):
            x_list_2.append(data_std[0:length])
            data_std = data_std[length:]

        return x_list_2

def split(all_x, all_y, percentage1, percentage2):
    #takes in all x and y features, and a decimal percentage, and returns the splitted versions
    total = len(all_x)
    fraction1 = round(float(len(all_x) * percentage1))
    fraction2 = round(float(len(all_x) * percentage2))
    sum = fraction1+fraction2

    train_x = all_x[:total-sum] #list of lists of floats
    val_x = all_x[total-sum:total-sum+fraction1]
    test_x = all_x[total-sum+fraction1:]  

    train_y = all_y[:total-sum] #list of integers
    val_y = all_y[total-sum:total-sum+fraction1]
    test_y = all_y[total-sum+fraction1:]
    return train_x, val_x, test_x, train_y, val_y, test_y



def array_and_reshape(train_x, val_x, test_x, train_y, val_y, test_y, length):
    #takes in the x and y splits and reshapes them to be correct
    train_x = np.array(train_x)
    val_x = np.array(val_x)
    test_x = np.array(test_x)
    train_y = np.array(train_y)
    val_y = np.array(val_y)
    test_y = np.array(test_y)

    train_x = np.reshape(train_x, (len(train_x), length,1))
    val_x = np.reshape(val_x, (len(val_x), length,1))
    test_x = np.reshape(test_x, (len(test_x), length,1))
    train_y = np.reshape(train_y, (len(train_y), 1,1))
    val_y = np.reshape(val_y, (len(val_y), 1,1))
    test_y = np.reshape(test_y, (len(test_y), 1,1))

    return train_x, val_x, test_x, train_y, val_y, test_y


def duplicate(train_x, test_x, train_y, test_y, times):
    #takes in the feature lists and essentially doubles them - appends them to oneself
    train_x = append_itself(train_x, times)
    test_x = append_itself(test_x, times)
    train_y = append_itself(train_y, times)
    test_y = append_itself(test_y, times)

    return train_x, test_x, train_y, test_y


def append_itself(lists, times):
    new_list = []
    for length in range(times):
        try:
            for example in lists:
                new_list.append(example)
        except:
            new_list.append(lists)
    return new_list


def plot(x_list, y_list, num):
    #plots a list based on a list of lists and its index
    h=num
    plt.plot(x_list[h], color='magenta', marker='o',mfc='pink' ) #plot the data
    # plt.xticks(range(0,len(train_x[0])+1, 1)) #set the tick frequency on x-axis

    plt.ylabel('data') #set the label for y axis
    plt.title("Stutter/No Stutter") #set the title of the graph
    plt.show() #display the graph


def to_pickle(train_x, val_x, test_x, train_y, val_y, test_y):
    import pickle
    with open('Pickles/train_x.ob', 'wb') as fp:
        pickle.dump(train_x, fp)
    with open('Pickles/val_x.ob', 'wb') as fp:
        pickle.dump(val_x, fp)
    with open('Pickles/test_x.ob', 'wb') as fp:
        pickle.dump(test_x, fp)
    with open('Pickles/train_y.ob', 'wb') as fp:
        pickle.dump(train_y, fp)
    with open('Pickles/val_y.ob', 'wb') as fp:
        pickle.dump(val_y, fp)
    with open('Pickles/test_y.ob', 'wb') as fp:
        pickle.dump(test_y, fp)


def scaler_to_pickle(scaler):
    import pickle
    with open('Pickles/scaler.ob', 'wb') as fp:
        pickle.dump(scaler, fp)


def move_test(test_x, test_y, names):
    total = len(names)
    total2 = len(test_x)
    fraction1 = round(float(total) * 0.15)
    names = names[total-fraction1:]

    for index in range(len(test_x)):

        if test_y[index] == 0:
            #move to non stutter
            src = os.path.join("StutterData", "NoStutter2", names[index])
            dst = os.path.join("StutterData", "test", "NoStutter_test")
            shutil.copy(src, dst)
        else:
            src = os.path.join("StutterData", "Stutter2", names[index])
            dst = os.path.join("StutterData", "test", "Stutter_test")
            shutil.copy(src, dst)
    pass
