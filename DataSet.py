import numpy as np
import os, re
import nltk
# nltk.download('averaged_perceptron_tagger')
from collections import Counter
from nltk.corpus import stopwords
from tqdm import tqdm

def loadDataSet():
    root = 'E:/TextC/20_newsgroups'
    dirpath = os.listdir(root)
    Doclist = []
    ClassVec = []
    stop_words1 = []
    with open('Data/stopwords_official.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            stop_words1.append(line)
    Part_of_speech = ['CD', 'FW', 'JJ', 'NN']
    tags = set(Part_of_speech)
    print('start loading and preprocessing data:')
    for i, _class in tqdm(enumerate(dirpath)):
        root2 = f'{root}/{_class}'
        idlist = os.listdir(root2)
        # if split == 'train':
        #     idlist = idlist[:800]
        # elif split == 'val':
        #     idlist = idlist[800:900]
        # elif split == 'test':
        #     idlist = idlist[900:]
        for id in idlist:
            filename = f'{root2}/{id}'
            with open(filename, encoding='utf-8', errors='ignore') as f:
                bigString = f.read()
                token_list = re.compile(r'\b[a-zA-Z]+\b', re.I).findall(bigString)  # 使用正则表达式匹配出非字母、非数字
                token_list = [tok.lower() for tok in token_list if len(tok) > 1]  # 去除一些太短的字符
                porter = nltk.PorterStemmer()
                token_list = [porter.stem(t) for t in token_list]  # 提取词干
                stop_words = set(stop_words1)
                token_list = [tok for tok in token_list if tok not in stop_words]  # 去除停用词
                pos_tags = nltk.pos_tag(token_list)
                token_list = [word for word, pos in pos_tags if pos in tags]
                Doclist.append(token_list)
                ClassVec.append(i)

    return Doclist, ClassVec

def loadDoc():

    filename = f'E:/TextC/20_newsgroups/alt.atheism/49960'
    with open(filename, encoding='utf-8', errors='ignore') as f:
        bigString = f.read()
        token_list = re.compile(r'\b[a-zA-Z]+\b', re.I).findall(bigString)  # 使用正则表达式匹配出非字母、非数字
        token_list = [tok.lower() for tok in token_list if len(tok) > 1]  # 去除一些太短的字符
        web_tags = ['href', 'http', 'https', 'www']
        token_list = [tok for tok in token_list if tok not in web_tags]  # 去除网页标签
        stop_words = set(stopwords.words('english'))
        token_list = [tok for tok in token_list if tok not in stop_words]  # 去除停用词
        porter = nltk.PorterStemmer()
        token_list = [porter.stem(t) for t in token_list]  # 提取词干

    return token_list

def createVocabList(DocList):
    VocabList = set()
    for Doc in tqdm(DocList):
        VocabList = VocabList | set(Doc)

    with open('Data/LowFreq_words.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line in VocabList:
                VocabList.remove(line)

    return list(VocabList)

def Doc2Vec(VocabList, Doc):
    Vec = np.zeros(len(VocabList))
    Doc = set(Doc)
    freq = Counter(Doc)
    for key, value in freq.items():
        if key in VocabList:
            Vec[VocabList.index(key)] += value
    return Vec

def Doc2Mat(VocabList, DocList):
    M = len(DocList)
    D = len(VocabList)

    P_mat = np.zeros((M, D))
    for i in tqdm(range(M)):
        freq = Counter(DocList[i])
        for key, value in freq.items():
            if key in VocabList:
                P_mat[i][VocabList.index(key)] += value
    return P_mat

def find_common_highFreq(DocList, ClassVec):
    n_class = len(set(ClassVec))
    HighFreq_set = set()
    for i in tqdm(range(n_class)):
        p_vec = []
        class_index = list(np.where(ClassVec == i))[0]
        for id in class_index:
            p_vec += DocList[id]
        freq = Counter(p_vec)
        # f = open(f'HIgh_Freq.txt', "a")
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)  # 按值从大到小排序
        hign_freq_i = []
        for idx in range(len(freq[:200])):
            key, value = freq[idx]
            hign_freq_i.append(key)
        if i == 0:
            HighFreq_set = set(hign_freq_i)
        else:
            HighFreq_set = HighFreq_set & set(hign_freq_i)
    with open('Data/High_Freq.txt', 'w') as f:
        for word in HighFreq_set:
            f.write(f'{word} ')
            f.write('\n')

    return HighFreq_set

def find_low_freq(DocList, ClassVec):
    n_class = len(set(ClassVec))
    LowFreq_set = set()
    for i in tqdm(range(n_class)):
        p_vec = []
        class_index = list(np.where(ClassVec == i))[0]
        for id in class_index:
            p_vec += DocList[id]
        freq = Counter(p_vec)
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=False)  # 按值从小到大排序
        low_freq_i = []
        for idx in range(len(freq)):
            key, value = freq[idx]
            if value < 3:
                low_freq_i.append(key)
        if i == 0:
            LowFreq_set = set(low_freq_i)
        else:
            LowFreq_set = LowFreq_set | set(low_freq_i)
    # print(LowFreq_set)
    with open('Data/LowFreq_words.txt', 'w', encoding='utf-8') as f:
        for line in list(LowFreq_set):
            f.write(f'{line}\n')

    return LowFreq_set