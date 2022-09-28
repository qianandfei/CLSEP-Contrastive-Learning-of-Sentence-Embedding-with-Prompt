# -*- coding: utf-8 -*-
# @Time    : 2021/9/25 16:04
# @Author  : wangqian

import json
import string

from zhon import hanzi


# 除去标点符号
def getPuncs():
    return set(hanzi.punctuation + string.punctuation)
Puncs=getPuncs()

# 此处处理停用词
def getStopWords():
    return set(' ')

def getNoneCh():
    return set(hanzi.punctuation + string.punctuation+
               string.digits+string.ascii_letters)
NoneCh=getNoneCh()

#统计一句话中非中文字符个数和比例，用来筛选那些垃圾句子
def CalNumOfNoneCh(text):
    if len(text)==0:
        return 0,0
    num=sum([i in NoneCh for i in text])
    return num,num/len(text)

def readFromJsonFile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def writeToJsonFile(path: str, obj, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=indent))

#有些大型json文件，是每一行自成json格式，按行写入的
def readFromJsonFileForLine(path: str):
    with open(path, "r", encoding="utf-8") as f:
        res=[]
        for i in f.readlines():
            data=i.strip()
            if len(data)!=0:
                res.append(json.loads(data))
        return res


def readFromJsonFile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def readLine(path: str):
    with open(path, "r", encoding="utf-8") as f:
        res=[]
        for i in f.readlines():
            data=i.strip()
            if len(data)!=0:
                res.append(data)
        return res

#读取序列标注类型数据，每一行有n列，每一列代表一种意义，空行作为分割，忽略多余空行
def readForSL(data_dir:str, n_clo:int):
    """读取数据"""
    all_data_lists=[[] for i in range(n_clo)]
    with open(data_dir, 'r', encoding='utf-8') as f:
        # 每段文本用空行隔开
        one_data_lists=[[] for i in range(n_clo)]
        for line in f:
            line = line.strip()
            if len(line) > 0:
                temp = line.split('\t')
                assert len(temp)==n_clo
                for i in range(n_clo):
                    one_data_lists[i].append(temp[i])
            elif len(one_data_lists[0]) != 0:#读到了段间空行
                for i in range(n_clo):
                    all_data_lists[i].append(one_data_lists[i])
                for i in range(n_clo):
                    one_data_lists[i]=[]
        if len(one_data_lists[0]) != 0:  # 防止最后一条数据无多余空行
            for i in range(n_clo):
                    all_data_lists[i].append(one_data_lists[i])
    return all_data_lists

#适用于需要不断追加写入的json文件，f为a+模式打开
#每写一次都打开关闭文件一次，可确保能基本实时保存，但效率低
def appendJsonLine(path: str, obj):
    with open(path, "a+", encoding="utf-8") as f:
        f.write(json.dumps(obj,ensure_ascii=False)+'\n')

