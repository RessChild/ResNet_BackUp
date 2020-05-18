import os
import time
from abc import abstractmethod
import tensorflow.compat.v1 as tf
from learning.utils import plot_learning_curve


class Optimizer(object):
    """경사 하강 러닝 알고리즘 기반 optimizer의 베이스 클래스."""

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        optimizer 생성자.
        :param model: ConvNet, 학습할 모델.
        :param train_set: DataSet, 학습에 사용할 학습 데이터셋.
        :param evaluator: Evaluator, 학습 수행 과정에서 성능 평가에 사용할 evaluator.
        :param val_set: DataSet, 검증 데이터셋, 주어지지 않은 경우 None으로 남겨둘 수 있음.
        :param kwargs: dict, 학습 관련 하이퍼파라미터로 구성된 추가 인자.
            - batch_size: int, 각 반복 회차에서의 미니배치 크기.
            - num_epochs: int, 총 epoch 수.
            - init_learning_rate: float, 학습률 초깃값.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        # 학습 관련 하이퍼파라미터
        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 320)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.01)

        self.learning_rate_placeholder = tf.placeholder(tf.float32)    # Placeholder for current learning rate
        self.optimize = self._optimize_op()

        self._reset()

    def _reset(self):
        """일부 변수를 재설정."""
        self.curr_epoch = 1
        self.num_bad_epochs = 0    # 'bad epochs' 수: 성능 향상이 연속적으로 이루어지지 않은 epochs 수.
        self.best_score = self.evaluator.worst_score   # 최저 성능 점수로, 현 최고 점수를 초기화함.
        self.curr_learning_rate = self.init_learning_rate    # 현 학습률 값

    @abstractmethod
    def _optimize_op(self, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.Optimizer.minimize Op.
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음.
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        고유의 학습률 스케줄링 방법에 따라, (필요한 경우) 매 epoch마다 현 학습률 값을 업데이트함.
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음.
        """
        pass

    def _step(self, sess, **kwargs):
        """
        경사 하강 업데이트를 1회 수행하며, 관련된 값을 반환함.
        해당 함수를 외부에서 임의로 호출할 수 없음.
        :param sess: tf.Session.
        :param kwargs: dict, 학습 관련 하이퍼파라미터로 구성된 추가 인자.
            - augment_train: bool, 학습 과정에서 데이터 증강을 수행할지 여부.
        :return loss: float, 1회 반복 회차 결과 손실 함숫값.
                y_true: np.ndarray, 학습 데이터셋의 실제 레이블.
                y_pred: np.ndarray, 모델이 반환한 예측 레이블.
        """
        augment_train = kwargs.pop('augment_train', True)

        # 미니배치 하나를 추출함
        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True,
                                              augment=augment_train, is_train=True)

        # 손실 함숫값을 계산하고, 모델 업데이트를 수행함
        _, loss, y_pred = \
            sess.run([self.optimize, self.model.loss, self.model.pred],
                     feed_dict={self.model.X: X, self.model.y: y_true,
                                self.model.is_train: True,
                                self.learning_rate_placeholder: self.curr_learning_rate})

        return loss, y_true, y_pred

    def train(self, sess, save_dir='/tmp', details=False, verbose=True, **kwargs):
        """
        optimizer를 실행하고, 모델을 학습함.
        :param sess: tf.Session.
        :param save_dir: str, 학습된 모델의 파라미터들을 저장할 디렉터리 경로.
        :param details: bool, 학습 결과 관련 구체적인 정보를, 학습 종료 후 반환할지 여부.
        :param verbose: bool, 학습 과정에서 구체적인 정보를 출력할지 여부.
        :param kwargs: dict, 학습 관련 하이퍼파라미터로 구성된 추가 인자.
        :return train_results: dict, 구체적인 학습 결과를 담은 dict.
        """
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())    # 전체 파라미터들을 초기화함

        train_results = dict()    # 학습 (및 검증) 결과 관련 정보를 포함하는 dict.
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        # 학습 루프를 실행함
        for i in range(num_steps):
            # 미니배치 하나로부터 경사 하강 업데이트를 1회 수행함
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)

            # 매 epoch의 말미에서, 성능 평가를 수행함
            if (i+1) % num_steps_per_epoch == 0:
                # 학습 데이터셋으로부터 추출한 현재의 미니배치에 대하여 모델의 예측 성능을 평가함
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)

                # 검증 데이터셋이 처음부터 주어진 경우, 이를 사용하여 모델 성능을 평가함
                if self.val_set is not None:
                    # 검증 데이터셋을 사용하여 모델 성능을 평가함
                    eval_y_pred = self.model.predict(sess, self.val_set, verbose=False, **kwargs)
                    eval_score = self.evaluator.score(self.val_set.labels, eval_y_pred)
                    eval_scores.append(eval_score)

                    if verbose:
                        # 중간 결과를 출력함
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))
                        # 중간 결과를 플롯팅함
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = eval_score

                # 그렇지 않은 경우, 단순히 미니배치에 대한 결과를 사용하여 모델 성능을 평가함
                else:
                    if verbose:
                        # 중간 결과를 출력함
                        print('[epoch {}]\tloss: {} |Train score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                        # 중간 결과를 플릇팅함
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = step_score

                # 현재의 성능 점수의 현재까지의 최고 성능 점수를 비교하고, 
                # 최고 성능 점수가 갱신된 경우 해당 성능을 발휘한 모델의 파라미터들을 저장함
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))    # 현재 모델의 파라미터들을 저장함
                else:
                    self.num_bad_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} score: {}'.format('evaluation' if eval else 'training',
                                             self.best_score))
        print('Done.')

        if details:
            # 학습 결과를 dict에 저장함
            train_results['step_losses'] = step_losses    # (num_iterations)
            train_results['step_scores'] = step_scores    # (num_epochs)
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores    # (num_epochs)

            return train_results


class MomentumOptimizer(Optimizer):
    """모멘텀 알고리즘을 포함한 경사 하강 optimizer 클래스."""

    def _optimize_op(self, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.MomentumOptimizer.minimize Op.
        :param kwargs: dict, optimizer의 추가 인자.
            - momentum: float, 모멘텀 계수.
        :return tf.Operation.
        """
        momentum = kwargs.pop('momentum', 0.9)

        update_vars = tf.trainable_variables()
        return tf.train.MomentumOptimizer(self.learning_rate_placeholder, momentum, use_nesterov=False)\
                .minimize(self.model.loss, var_list=update_vars)

    def _update_learning_rate(self, **kwargs):
        """
        성능 평가 점수 상에 개선이 없을 때, 현 학습률 값을 업데이트함.
        :param kwargs: dict, 학습률 스케줄링을 위한 추가 인자.
            - learning_rate_patience: int, 성능 향상이 연속적으로 이루어지지 않은 epochs 수가 
                                      해당 값을 초과할 경우, 학습률 값을 감소시킴.
            - learning_rate_decay: float, 학습률 업데이트 비율.
            - eps: float, 업데이트된 학습률 값과 기존 학습률 값 간의 차이가 해당 값보다 작을 경우,
                          학습률 업데이트를 취소함.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # 새 학습률 값과 기존 학습률 값 간의 차이가 eps보다 큰 경우에 한해서만 업데이트를 수행함
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_bad_epochs = 0
