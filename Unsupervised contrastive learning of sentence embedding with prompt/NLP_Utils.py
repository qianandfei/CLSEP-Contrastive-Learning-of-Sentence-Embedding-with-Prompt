# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 16:04
# @Author  : wangqian

import json
def readFromJsonFile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def writeToJsonFile(path: str, obj, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=indent))


def readLine(path: str):
    with open(path, "r", encoding="utf-8") as f:
        res=[]
        for i in f.readlines():
            data=i.strip()
            if len(data)!=0:
                res.append(data)
        return res



