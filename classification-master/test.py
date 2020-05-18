import os
import numpy as np
import tensorflow.compat.v1 as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.evaluators import AccuracyEvaluator as Evaluator


""" 1. 원본 데이터셋을 메모리에 로드함 """
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
test_dir = os.path.join(root_dir, 'test')

# 테스트 데이터셋을 로드함
X_test, y_test = dataset.read_asirra_subset(test_dir, one_hot=True)
test_set = dataset.DataSet(X_test, y_test)

# 중간 점검
print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


""" 2. 테스트를 위한 하이퍼파라미터 설정 """
hp_d = dict()
image_mean = np.load('/tmp/asirra_mean.npy')    # load mean image
hp_d['image_mean'] = image_mean

# FIXME: 테스트 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['augment_pred'] = True


""" 3. Graph 생성, 파라미터 로드, session 초기화 및 테스트 시작 """
# 초기화 
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, '/tmp/model.ckpt')   # 학습된 파라미터 로드 및 복원
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
