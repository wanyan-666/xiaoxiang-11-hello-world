# 交叉验证训练和测试模型
from sklearn.model_selection import StratifiedShuffleSplit
import codecs, gc
import keras.backend as K
import json
from keras.callbacks import Callback
import sys
sys.path.append("./model/")
from  mod1  import  *



def run_cv(nfold, data, y):
    kf = StratifiedShuffleSplit(n_splits=nfold, test_size=0.4, random_state=2020).split(data, y)

    for train_fold, test_fold in kf:
        X_train, X_valid = data[train_fold], data[test_fold]
        y_trian, y_valid = y[train_fold], y[test_fold]

        print("初始化类...")
        model = UserBaseAttributes()

        # 模型训练
        print("训练模型...")
        model.fit(X_train, y_trian)

        # 返回预测准确率
        print(model.score(X_valid, y_valid))

        del model
        gc.collect()  # 清理内存
        # break

    return


# 神经网络配置
class_num = 3
max_features = 35450  # 259884
maxlen = 500
batch_size = 64
embedding_dims = 50  # 300
epochs = 8
word_to_id = mk_dict()
'''
from gensim.models import KeyedVectors
print('读取预训练的词向量')
path_word2vec = './model/sgns.zhihu.bigram.bz2'  
word2vec = KeyedVectors.load_word2vec_format(path_word2vec, binary=False)
# path_word2vec = './model/baike_26g_news_13g_novel_229g.model'
# word2vec = Word2Vec.load(path_word2vec)

embeddings_matrix ,word_to_id = build_matrix(word2vec)
'''



def run_cv2(nfold, data, y):
    kf = StratifiedShuffleSplit(n_splits=nfold, test_size=0.4, random_state=2020).split(data, y)

    for train_fold, test_fold in kf:
        X_train, X_valid = data[train_fold], data[test_fold]
        y_trian, y_valid = y[train_fold], y[test_fold]

        # 对文本的词id和类别id进行编码
        x_train = encode_sentences(X_train, word_to_id)
        y_train = to_categorical(y_trian)

        x_test = encode_sentences(X_valid, word_to_id)
        y_test = to_categorical(y_valid)

        print('对序列做padding，保证是 samples*timestep 的维度')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        print('构建模型...')
        # metrics = Metrics_new()
        # metrics = Metrics(x_test, y_test)
        metrics = Metrics_new2(valid_data = (x_test, y_test))
        model = TextCNN(maxlen, max_features, embedding_dims, class_num).get_model()  # embeddings_matrix
        model.compile('adam', 'categorical_crossentropy', metrics=['acc'])

        print('训练...')
        # 设定callbacks回调函数
        my_callbacks = [metrics,
            ModelCheckpoint('./data/cnn_model.h5',monitor='val_f1', verbose=1),
            EarlyStopping(monitor='val_f1', patience=4, mode='max')]

        # fit拟合数据  val_acc
        # print(x_train[:2])
        # print(y_train[:2])
        print(type(x_train), type(y_train), type(x_test), type(y_test))
        print(batch_size)
        print(epochs)

        print('训练2...')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=my_callbacks,
                            validation_data=(x_test, y_test))

        del model
        gc.collect()  # 清理内存
        K.clear_session()  # clear_session就是清除一个session

    return history




import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


path = "./data/ori_data/"
trian_file = path+"train.json"
train = pd.read_json(trian_file)


df = train[train.Gender != 0]
df_new = df[["Gender","Query_List"]]

train_data = np.array(df_new.values.tolist())

target = train_data[:,0]
text = train_data[:,-1]

y_train = target
x_train  = []
for line in text:
    str = ''
    for char in line:
        str0 = " ".join(char)
        str = str + " "+ str0

    x_train.append(str)


train_data = np.array(x_train)
y = np.array(y_train,dtype=np.int)


run_cv(5, train_data, y)
print("end.....")


new_train = []
for line in text:
    lst = [word for char in line for word in char]
    new_train.append(lst)

new_train = np.array(new_train)
new_y = np.array(target,dtype=np.int)

history = run_cv2(3, new_train, new_y)

print("end2.....")

