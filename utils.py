import os
import random
from subprocess import list2cmdline
from unittest import result
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json
import torch
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

classifications = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]




'''
description: 对列表进行分割
param {*} full_list 待分割的列表
param {*} shuffle 是否打乱
param {*} ratio1 第一个分割出的列表的比例
param {*} ratio2 第二个分割出的列表的比例
return {*} 分割出的第一个列表，分割出的第二个列表，分割出的第三个列表
'''
def ran_split(full_list,shuffle=False,ratio1=0.8,ratio2=0.1):
	n_total = len(full_list)
	offset1 = int(n_total * ratio1)
	offset2 = int(n_total * ratio2) + offset1
	if n_total == 0 or offset1 < 1:
		return [], full_list
	if shuffle:
		random.shuffle(full_list)   # 打乱排序
	return full_list[:offset1], full_list[offset1:offset2], full_list[offset2:]

'''
description: 用于读取xml文件
param {*} root 文件根目录
param {*} paths 文件路径
return {*} 路径列表，标注列表
'''
def read_xml(root_path, paths, image_root_paths):
    path_list = []
    annotation_list = []
    bar = tqdm(paths)
    for p in bar:
        tree = ET.parse(root_path + "\\" + p)
        root = tree.getroot()
        for child in root:
            if(child.tag == 'filename'):
                if child.text in image_root_paths:
                    path_list.append(child.text)
                else:
                    break
            elif(child.tag == 'object'):
                for c in child:
                    if(c.tag == 'name'):
                        annotation_list.append(c.text)
                        break
                break
    return path_list, annotation_list


def write_data_path(filename, data_paths, data_paths_labels):
    with open("classification.json", "r") as f:
        classification = json.load(f)
    data_paths = tqdm(data_paths)
    data_paths_labels = tqdm(data_paths_labels)
    with open(filename + "-path.txt", 'w') as file1:
        file1.truncate(0)
        for path in data_paths:
            
                file1.write(path + '\n')
        print("[Success] Write paths of data in file {0}".format(filename + "-path.txt"))
    with open(filename + "-anno.txt", 'w') as file2:
        file2.truncate(0)
        for annotation in data_paths_labels:
            label = classification[annotation]
            file2.write(label + '\n')
        print("[Success] Write annotations of data in file {0}".format(filename + "-anno.txt"))
        
        

def read_file(filename, form = None):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            if(form == "int"):
                data.append(int(line.strip()))
            elif(form == "float"):
                data.append(float(line.strip()))
            else:
                data.append(line.strip())
    return data


def write_data(filename, list):
    with open(filename, 'w') as file:
        file.truncate(0)
        for data in list:
            file.write(str(data) + '\n')
        print("[Success] Write {0} lines of data in file {1}".format(len(list), filename + "-path.txt"))
        
@torch.no_grad()
def evaluate(model, data_loader, device, epoch, print_info):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    if print_info:
        data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels, filePath = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        if print_info:
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


# delete


@torch.no_grad()
def evaluate_save(model, data_loader, device, epoch, save = False):
    
    paths = []
    ls = []
    pres = []
    acs = []
    
    
    
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels, filePath = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        
        
        accurates = torch.eq(pred_classes, labels.to(device))

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        
        if save:
            for index, path in enumerate(filePath):
                label = labels[index]
                pre = pred_classes[index].cpu()
                accurate = accurates[index].cpu()
                pres.append(pre)
                acs.append(accurate)
                ls.append(label)
                paths.append(path)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, paths, ls, pres, acs


def get_weight_path(root, num):
    return root + str(num) + ".pth"


def get_class_from_path(path):
    """根据文件路径返回类别

    Args:
        path (_string_): 文件路径

    Returns:
        _string_: 类别
    """
    with open("../cifar10/classification.json", "r") as f:
        classification = json.load(f)
    path = Path(path)
    return classification[str(path.parts[3])]


def get_class_from_path2(path):
    """根据文件路径返回类别

    Args:
        path (_string_): 文件路径

    Returns:
        _string_: 类别
    """
    path = Path(path)
    return str(path.parts[3])


def tracin_get(a, b):
    """ dot product between two lists"""
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])
    # breakpoint()


def get_gradient(grads, model):
    """
    pick the gradients by name.
    """
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())]

def get_sorted_index(list, ration = 1.0):
    """对list进行排序，返回下表

    Args:
        list (_list_): 待排序的数组
        ration (float, optional): 返回前ration%个. Defaults to 1.0.

    Returns:
        _list_: 从大到小的下表组成的list，取前ration%个
    """
    index  = sorted(range(len(list)), key=lambda k: list[k], reverse=True)
    return index[:int(len(list)*ration)]


def statistic_histogram(data, name):
    return 0


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for index, row in enumerate(plots):
        if index == 0:
            y.append((row[2])) 
            x.append((row[1]))
        else:
            y.append(float(row[2])) 
            x.append(float(row[1]))
    return x ,y


def statistic_histogram2(num1, num2):
    data1 = read_file("../tracin_file/checkpoint" + str(num1) + "_0.3.txt")
    data2 = read_file("../tracin_file/checkpoint" + str(num2) + "_0.3.txt")
    nums1 = [0,0,0,0,0,0,0,0,0,0]
    for a in data1:
        nums1[classifications.index(get_class_from_path2(a))] = nums1[classifications.index(get_class_from_path2(a))] + 1
        
    nums2 = [0,0,0,0,0,0,0,0,0,0]
    for a in data2:
        nums2[classifications.index(get_class_from_path2(a))] = nums2[classifications.index(get_class_from_path2(a))] + 1

    plt.figure(figsize=(20,10), dpi=80)
    plt.style.use('ggplot')
    plt.xlabel('Class')
    plt.ylabel('Number')

    ax1 = plt.subplot(221)
    ax1.set_title('The statistics of epoch {}'.format(str(num1)))
    ax1.bar(
        x = classifications, ## 设置x轴内容
        height = nums1,  ## 设置y轴内容
    )

    ax2 = plt.subplot(222)
    ax2.set_title('The statistics of epoch {}'.format(str(num2)))
    ax2.bar(
        x = classifications, ## 设置x轴内容
        height = nums2,  ## 设置y轴内容
    )
    
    
def statistic_histogram(txt):
    data1 = read_file(txt)
    nums1 = [0,0,0,0,0,0,0,0,0,0]
    for a in data1:
        nums1[classifications.index(get_class_from_path2(a))] = nums1[classifications.index(get_class_from_path2(a))] + 1
        
    
    plt.figure(figsize=(10,5), dpi=80)
    plt.style.use('ggplot')
    plt.xlabel('Class')
    plt.ylabel('Number')
    
    plt.title('The statistics of {}'.format(str(txt)))
    plt.bar(
        x = classifications, ## 设置x轴内容
        height = nums1,  ## 设置y轴内容
    )

def show_lines(x, y):
    colors = ['darkred','burlywood','cyan','darkgreen','darkviolet','red','yellow']
    plt.figure(figsize=(15,10), dpi=80)
    for index, data in enumerate(x):
        plt.plot(data[1:], y[index][1:], color=colors[index], label='Default')
        plt.xlabel('Steps',fontsize=20)
        plt.ylabel('Acc',fontsize=20)
    plt.show()
   
   
   
import yaml 
    
def read_yaml(path):
    f = open(path, encoding='utf-8')

    data = yaml.load(f.read(), Loader=yaml.FullLoader)

    f.close()
    return data

def count(list):
    count = 0
    for data in list:
        if data == False:
            count = count + 1
    return count



def cal_wrong(data, batch, ration1, ration2):
    before_batch_nums = []
    after_batch_nums = []
    wrong = []
    path = []
    for index, d in enumerate(data):
        path.append(d['path'])
        before_batch = d['acc'][:batch]
        after_batch = d['acc'][batch:]
        before_batch_num = count(before_batch)
        after_batch_num = count(after_batch)
        before_batch_nums.append(before_batch_num)
        after_batch_nums.append(after_batch_num)
        wrong.append(round(before_batch_num * ration1 + after_batch_num * ration2, 2))
        
    return wrong, before_batch_nums, after_batch_nums, path



def sort_list(list1, list2, ration):
    list1_sorted = []
    list2_sorted = []
    for index in get_sorted_index(list1, ration):
        list1_sorted.append(list1[index])
        list2_sorted.append(list2[index])
        
    return list1_sorted, list2_sorted


def split_easy_hard(list, hard_ration, easy_ration, classify = True):
    if classify:
        classifications = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

        classes = []

        for c in classifications:
            c = []
            classes.append(c)

        for data in list:
            classes[int(classifications.index(get_class_from_path2(data)))].append(data)
            
        r_hard = []
        r_easy = []
        
        for c in classes:
            for c2 in c[:int(len(list)*hard_ration/len(classifications))]:
                r_hard.append(c2)
            for c2 in c[-int(len(list)*easy_ration/len(classifications)):]:
                r_easy.append(c2)
                
        return r_hard, r_easy
    
    
    
    else:
        return list[:int(len(list)*hard_ration)], list[-int(len(list)*easy_ration):]
    




def split_easy_hard_num(list, num_head, num_rear, rate):
    classifications = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    classes = []

    for c in classifications:
        classes.append([])

    for data in list:
        classes[int(classifications.index(get_class_from_path2(data)))].append(data)
            
    data_splited = []
        
    for c in classes:
        for c2 in c[int(num_head/10):int(num_rear/10)]:
                data_splited.append(c2)
        
    
        
    return random.sample(data_splited, rate)
    


    
def draw_hist(path):
    class_sort = []

    for index, p in enumerate(path):
        class_sort.append(get_class_from_path2(p))

    n, bins, patches = plt.hist(class_sort, 10, facecolor='g')
    plt.close()
    classifications = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    plt.figure(figsize=(10,5), dpi=80)
    plt.style.use('ggplot')
    plt.xlabel('Class')
    plt.ylabel('Number')

        
    plt.title('Hist of data')
    plt.bar(
        x = classifications, ## 设置x轴内容
        height = n,  ## 设置y轴内容
    )

    plt.show()
    
    
    
def split_list(list, head_ration, rear_ration, classify = True):
    if classify:
        classifications = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

        classes = []

        for c in classifications:
            c = []
            classes.append(c)

        for data in list:
            classes[int(classifications.index(get_class_from_path2(data)))].append(data)
            
        result_list = []
        
        for c in classes:
            for c2 in c[int(len(list)*head_ration/len(classifications)):int(len(list)*rear_ration/len(classifications))]:
                result_list.append(c2)
                
        return result_list
    
    
    
    else:
        return list[int(len(list)*head_ration):int(len(list)*rear_ration)]
    
    
    
def get_description(train_data, segments):
    description = "*TOTAL:" + str(len(train_data)) + "*"
    for i, segment in enumerate(segments):
        description += "[" + str(segment[0]) + "--" + str(segment[1]) + "]->"  + str(segment[2])
        if i + 1 != len(segments):
            description += "&"
            
    return description