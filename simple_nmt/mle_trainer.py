import numpy as np
import torch

from torch import optim
import torch.nn.utils as torch_utils
from ignite.engine import Engine, Events

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MaximumLikelihoodEstimationTrainer():

    def __init__(self, config):
        self.config = config

    @staticmethod #mle에 맞는 step함수 작성.
    def step(engine, mini_batch): #for문을 없앴다. train 부분. 매 미니배치마다 이부분이 자동으로 실행됨. for문 쓸 필요 없음.
        from utils import get_grad_norm, get_parameter_norm

        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train() #train 훈련 x. evaluation 모드와 training 모드를 홨다갔다 하는것. train모드로 가라.
        engine.optimizer.zero_grad() #zero gradient 초기화.

        # Raw target variable has both BOS and EOS token. 
        # The output of sequence-to-sequence does not have BOS token. 
        # Thus, remove BOS token for reference.
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:] #y의 경우에는 bos와 eos가 둘다 들어있다. y햇과 비교할거니까 bos를 떼었고.
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        # Take feed-forward
        # Similar as before, the input of decoder does not have EOS token.
        # Thus, remove EOS token for decoder input.
        y_hat = engine.model(x, mini_batch.tgt[0][:, :-1]) #eos 떼어낸 것을 넣어준다.
        #log softmax 결과값이니까
        # |y_hat| = (batch_size, length, output_size)

        loss = engine.crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                           y.contiguous().view(-1)
                           )
        loss.div(y.size(0)).backward()
        word_count = int(mini_batch.tgt[1].sum())

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # In orther to avoid gradient exploding, we apply gradient clipping.
        torch_utils.clip_grad_norm_(
            engine.model.parameters(),
            engine.config.max_grad_norm,
        )
        # Take a step of gradient descent.
        engine.optimizer.step()

        return float(loss / word_count), p_norm, g_norm #ignite에서 다른 부분.
        #loss 인데 글자수로 나눈 로스와, parameter norom 과 gradient norm 반환. 화면에 뿌리는걸 하기 위해서 준거임. logger를 위해. attach와 연결됨.

    @staticmethod
    def validate(engine, mini_batch):
        from utils import get_grad_norm, get_parameter_norm

        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, n_classes)
            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1),
            )
            word_count = int(mini_batch.tgt[1].sum())

        return float(loss / word_count)

    @staticmethod
    def attach(trainer, evaluator, verbose=VERBOSE_BATCH_WISE): 
        from ignite.engine import Events
        from ignite.metrics import RunningAverage
        from ignite.contrib.handlers.tqdm_logger import ProgressBar

        # step함수 마지막 리턴과 연결됨.
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss') 
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, '|param|')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, '|g_param|')

        if verbose >= VERBOSE_BATCH_WISE: #progress bar 찍어줌.
            pbar = ProgressBar(bar_format=None, ncols=120) 
            pbar.attach(trainer, ['|param|', '|g_param|', 'loss'])

        if verbose >= VERBOSE_EPOCH_WISE: #progress bar 안쓰고, 보여주고 싶을때
            @trainer.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    np.exp(avg_loss),
                ))

        RunningAverage(output_transform=lambda x: x).attach(evaluator, 'loss')

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(evaluator, ['loss'])

        if verbose >= VERBOSE_EPOCH_WISE:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']
                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))

    @staticmethod
    def check_best(engine):
        from copy import deepcopy

        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % (train_engine.epoch_idx),
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss)
                                                   )
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )

    def train(self, model, crit, optimizer, train_loader, valid_loader, src_vocab, tgt_vocab, n_epochs): #강사분이 구현한 부분..
        #엔진은 두개 있다. train용, evaluator용
        trainer = Engine(self.step)
        trainer.config = self.config
        trainer.model, trainer.crit, trainer.optimizer = model, crit, optimizer
        trainer.epoch_idx = 0

        evaluator = Engine(self.validate)
        evaluator.config = self.config
        evaluator.model, evaluator.crit = model, crit
        evaluator.best_loss = np.inf

        self.attach(trainer, evaluator, verbose=self.config.verbose) 

        @trainer.on(Events.EPOCH_COMPLETED)
        def epoch_cnt(engine): #arg를 engine밖에 못함
            engine.epoch_idx += 1

        def run_validation(engine, evaluator, valid_loader): #engine 보다 많은 arg를 줄 수 있다. handler를 add해서 사용한것임.
            evaluator.run(valid_loader, max_epochs=1)

        trainer.add_event_handler( # 한에폭이 끝났을 때 evaluate를 한 번 돌려라. traniner의 한 에폭이 끝나면 evaluator를 한번 돌리는것.
            Events.EPOCH_COMPLETED, run_validation, evaluator, valid_loader
        )
        evaluator.add_event_handler( # evaluator 이벤트 등록.
            Events.EPOCH_COMPLETED, self.check_best #현재 로스보다 더 좋은 로스인지 체크하는 함수. check_best
        )
        evaluator.add_event_handler( # 같은 이벤트는 등록한 순서대로 실행됨. 먼저 선언된게 먼저 수행됨.
            Events.EPOCH_COMPLETED,
            self.save_model, #validation 끝났으니까 model save..
            trainer,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        trainer.run(train_loader, max_epochs=n_epochs) #내가 지정한 에폭만큼 run gka. gks dpvhrdl Rmxskaus 겨ㅜ_ㅍ미ㅑㅇㅁ샤ㅐㅜd
        #매 mini batch마다 self.validate

        return model
