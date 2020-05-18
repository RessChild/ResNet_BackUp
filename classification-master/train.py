# http://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html
# https://github.com/sualab/asirra-dogs-cats-classification

import os
import numpy as np
import tensorflow.compat.v1 as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import AccuracyEvaluator as Evaluator


""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
# @@ 데이터 파일이 존재하는 경로
root_dir = os.path.join('./', 'datas', 'asirra')    # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# @@ 경로 추가
save_dir = os.path.join('./','result','testing') 

# 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

# 중간 점검
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


""" 2. 학습 수행 및 성능 평가를 위한 하이퍼파라미터 설정 """
hp_d = dict()
image_mean = train_set.images.mean(axis=(0, 1, 2))    # 평균 이미지
np.save('/tmp/asirra_mean.npy', image_mean)   # 평균 이미지를 저장
hp_d['image_mean'] = image_mean

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 300

hp_d['augment_train'] = True
hp_d['augment_pred'] = True

hp_d['init_learning_rate'] = 0.01
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: 정규화 관련 하이퍼파라미터
hp_d['weight_decay'] = 0.0005
hp_d['dropout_prob'] = 0.5

# FIXME: 성능 평가 관련 하이퍼파라미터
hp_d['score_threshold'] = 1e-4


""" 3. Graph 생성, session 초기화 및 학습 시작 """
# 초기화
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
train_results = optimizer.train(sess, save_dir=save_dir, details=True, verbose=True, **hp_d)
