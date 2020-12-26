import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
import jieba
import json


# 分析数据分布后：
# 每次搜索字符串最小值是1,最大是245,长度大于15的占比为 4.5% 。 只取大于1而且小于 15
# 搜索次数,最小值是49,最大是498,大于250次以上的占比为 7.3% 。  对于搜索次数大于250次以上的，只取后面250次搜索内容
def handle_data(lst):
    lst_new = []
    # 对于搜索次数大于250次以上的，只取后面250次搜索内容
    lst = lst[-250:] if len(lst) > 250 else lst
    for text2 in lst:
        str2 = ''.join(text2)
        # 单次搜索的字符串小于2或大于15,则舍弃
        if len(str2) < 2 or len(str2) > 15:
            continue
        # 分词后，词语长度小于2或者纯数值，则舍弃
        lst_word = [word for word in text2 if len(word.strip()) >= 2 or word in ['L', 'N', 'Q', 'W']]
        if len(lst_word) > 0:
            lst_new.append(lst_word)
    return lst_new


# 将疑问词、网址、数字和英文替换成 特定字母
def split_and_masked(text):
    Q1 = ["谁", "何", "什么", "哪儿", "哪里", "几时", "几", "多少", "怎么着", "怎么样", "怎么", "怎的", "怎样", "如何", "为什么", "怎", "吗", "呢", "吧",
          "啊"]

    p1 = re.compile("|".join(Q1))
    p2 = re.compile(
        "(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\,\'\/\\\+&amp;%$#_]*)?")
    p3 = re.compile("|".join(["\d+\\.\d+", "\d+"]))  # 数字
    p4 = re.compile("[a-zA-Z]+")  # 字母

    lst_new = []
    lst = text.split("\t")
    n = len(lst)
    for line in lst:
        if re.search(p2, line) is None: line = re.sub(p4, 'L', line)
        if re.search(p2, line) is None:  line = re.sub(p3, 'N', line)
        line = re.sub(p1, 'Q', line)
        line = re.sub(p2, 'W', line)

        Lst_cut = jieba.lcut(line)
        Lst_cut2 = [char for char in Lst_cut if
                    len(char) >= 2 or char in ['L', 'N', 'Q', 'W']]  # or char in ['L','N','Q','W']

        # str_cut = ' '.join(Lst_cut2)
        lst_new.append(Lst_cut2)

    lst_new = handle_data(lst_new)

    return lst_new


def build_vocab(data):
    """根据训练集构建词汇表，存储"""
    data_train = data

    all_data = {}

    for content in data_train:
        for line in content:
            for char in line:
                all_data = update_dic(char, all_data)

    dic_words = all_data.copy()
    # 剔除词频小于等于100 的词语
    for char, v in all_data.items():
        if v <= 100:
            del dic_words[char]
        else:
            continue

    # 制作key-values键值对，预留 20 个位置，以便其他特殊字符使用
    idx = 20
    for char, v in dic_words.items():
        dic_words[char] = idx
        idx += 1

    return dic_words

def update_dic(char, word2tf):
    # 制作key-values键值对，其中values为词频
    if word2tf.get(char) is None:
        word2tf[char] = 1
    else:
        word2tf[char] += 1
    return word2tf


path = "./data/ori_data/"
train = pd.read_csv(path+"train.csv", encoding="UTF-8", sep="###__###",header = None)
train.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']

train['Query_List']=train['Query_List'].apply(split_and_masked)

# 保存预处理好,带标签的训练数据
# 写入json
trian_file = path+"train.json"
train.to_json(trian_file)
# read_json = pd.read_json(trian_file)

test = pd.read_csv(path+"test.csv", encoding="UTF-8", sep="###__###",header = None)
test.columns = ['ID', 'Query_List']

test['Query_List']=test['Query_List'].apply(split_and_masked)

# 保存预处理好,带标签的训练数据
# 写入json
test_file = path+"test.json"
test.to_json(test_file)


# 制作词典，写入json
data = train['Query_List'].tolist()
dic = build_vocab(data)

vocab_file = path+"vocab.json"
with open(vocab_file, 'w+', encoding='utf-8') as f:
    f.write(json.dumps(dic, ensure_ascii=False))


