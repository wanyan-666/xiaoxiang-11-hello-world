from tensorflow.keras.models import load_model
import tensorflow as tf
import shutil
import  pandas as pd
import  numpy as np
import  json
import requests
import  os
import sys
sys.path.append("./model/")
from  mod1  import  *

# 模型的加载
path_model = "./model/"
Age_model = load_model(path_model+'Age_cnn_model.h5')
Gender_model = load_model(path_model+'Gender_cnn_model.h5')
Education_model = load_model(path_model+'Education_cnn_model.h5')
model_lst = [Age_model,Gender_model,Education_model]

def yaoqiu1(model_lst):
    path = "./data/ori_data/"
    trian_file = path+"train.json"
    train = pd.read_json(trian_file).head(1000)
    train_data = np.array(train.values.tolist())

    Query_List = train_data[:,-1]
    new_train = []
    for line in Query_List:
        lst = [word for char in line for word in char]
        new_train.append(lst)
    new_train = np.array(new_train)

    y_true = []
    for i in range(1,4):
        target = train_data[:,i]
        y = np.array(target,dtype=np.int)
        y_true.append(y)

    maxlen = 500
    batch_size = 64
    # 获取字典
    word_to_id = mk_dict()

    # 对文本的词id和类别id进行编码
    x_test = encode_sentences(new_train, word_to_id)

    print('对序列做padding，保证是 samples*timestep 的维度')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_test shape:', x_test.shape)

    acc_lst = []
    for m,y in zip(model_lst,y_true):
        pre = m.predict(x_test, batch_size=batch_size)
        acc_new = round(np.sum(pre.argmax(axis=1) == y) / y.shape[0], 2)
        acc_lst.append(acc_new)
    acc_res = round(np.mean(acc_lst),2)

    print("年龄的准确率：",acc_lst[0],"性别的准确率：",acc_lst[1],"学历的准确率：",acc_lst[2])
    print("总体正确率: ",acc_res)

def yaoqiu2(path_model,model_lst):
    # 模型导出
    label_lst = ['Age_CNN','Gender_CNN','Education_CNN']
    path_lst = []
    for p,m in zip(label_lst,model_lst):
        # 指定路径
        if os.path.exists(path_model+'Models/'+p+'/1'):
            shutil.rmtree(path_model+'Models/'+p+'/1')

        export_path = path_model+'Models/'+p+'/1'
        path_lst.append(export_path)
        # 导出tensorflow模型以便部署
        tf.saved_model.save(m, export_path)

    path = "./data/ori_data/"
    test_file = path+"test.json"
    test = pd.read_json(test_file).head(1000)
    test_data = np.array(test.values.tolist())

    Query_List = test_data[:,-1]
    new_test = []
    for line in Query_List:
        lst = [word for char in line for word in char]
        new_test.append(lst)

    maxlen = 500
    batch_size = 64
    # 获取字典
    word_to_id = mk_dict()

    # 对文本的词id和类别id进行编码
    text_seg = encode_sentences(new_test, word_to_id)

    print('对序列做padding，保证是 samples*timestep 的维度')
    text_input = sequence.pad_sequences(text_seg, maxlen=maxlen)
    print('x_test shape:', x_test.shape)

    text_input_1 = text_input[:3]
    data = json.dumps({"signature_name": "serving_default",
                       "instances": text_input_1.tolist()})
    headers = {"content-type": "application/json"}

    age_response = requests.post('http://localhost:8501/v1/models/Age:predict',
                                  data=data, headers=headers)
    Education_response = requests.post('http://localhost:8501/v1/models/Education:predict',
                                  data=data, headers=headers)
    Gender_response = requests.post('http://localhost:8501/v1/models/Gender:predict',
                                  data=data, headers=headers)
    #print(json.loads(json_response.text))
    print("predict Age:",age_response.text)
    print("predict Education:",Education_response.text)
    print("predict Gender:",Gender_response.text)


