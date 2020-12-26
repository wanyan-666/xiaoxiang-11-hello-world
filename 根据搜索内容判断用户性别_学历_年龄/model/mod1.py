import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_curve, auc

from joblib import dump, load
import  json

class UserBaseAttributes():
    # 成员函数
    def __init__(self, classifier=MultinomialNB()):
        #         self.classifier = RandomForestClassifier(n_estimators=15,random_state=2018)
        self.classifier = LogisticRegression(C=1.0, tol=1e-6, multi_class='multinomial', solver='newton-cg')
        # self.classifier = GradientBoostingClassifier(random_state=10)
        # ngram_range=(1,2),
        #         self.vectorizer = CountVectorizer(ngram_range=(1,3), max_features=3500, preprocessor=self._remove_noise,min_df =50)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, min_df=50)

    # 私有函数，数据清洗
    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        clean_text = re.sub(noise_pattern, "", document)
        return clean_text

    # 特征构建
    def features(self, X):
        return self.vectorizer.transform(X)

    # 拟合数据
    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    # 预估类别
    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    # 测试集评分
    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

    # 模型持久化存储
    def save_model(self, path):
        dump((self.classifier, self.vectorizer), path)

    # 模型加载
    def load_model(self, path):
        self.classifier, self.vectorizer = load(path)


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
import random
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec,KeyedVectors
from keras.callbacks import Callback
import tensorflow as tf
from tensorflow import keras

class TextCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims, class_num, last_activation='softmax'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self,embedding_matrix=None):
        input = Input((self.maxlen,))
        if embedding_matrix is None:
            print("AAAA")
            embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        else:
            print("BBBB")
            embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen, weights=[embedding_matrix])(input)

        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


def build_matrix(word2vec):
    ## 构造包含所有词语的 list，以及初始化 “词语-序号”字典 和 “词向量”矩阵
    vocab_list = [word for word, Vocab in word2vec.vocab.items()]# 存储 所有的 词语

    word_index = {" ": 0}# 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {} # 初始化`[word : vector]`字典

    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。

    embeddings_matrix = np.zeros((len(vocab_list) + 1, word2vec.vector_size))

    ## 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        # print(i)
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1 # 词语：序号
        word_vector[word] = word2vec[word] # 词语：词向量
        embeddings_matrix[i + 1] = word2vec[word]  # 词向量矩阵

    return embeddings_matrix,word_index

class Metrics_new2(Callback):
    def __init__(self, valid_data):
        super(Callback, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


def mk_dict():
    vocab_file = "./data/ori_data/vocab.json"
    with open(vocab_file, 'r', encoding="utf-8") as sf:
        word_to_id = json.load(sf)

    unk_index = 2  # 用来表达未知的字, 如果字典里查不到
    word_to_id["#UNK#"] = unk_index
    return word_to_id



def encode_cate(content, words):
    """将id表示的内容转换为文字"""
    return [(words[x] if x in words else words["#UNK#"]) for x in content]

def encode_sentences(contents, words):
    """将id表示的内容转换为文字"""
    return [encode_cate(x, words) for x in contents]

