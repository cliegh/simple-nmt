# from nltk.translate.bleu_score import sentence_bleu as score_func
from nltk.translate.gleu_score import sentence_gleu as score_func
#bleu나 gleu 중 골라서 쓰면 됨. 둘다 해 보고 성능이 더좋게 나오는것 써도 되고..

import torch

import torch.nn.utils as torch_utils
from ignite.engine import Engine, Events

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


from simple_nmt.mle_trainer import MaximumLikelihoodEstimationTrainer

#강화학습
class MinimumRiskTrainer(MaximumLikelihoodEstimationTrainer): # MaximumLikelihoodEstimationTrainer 상속 받음. train함수만 갖다씀. 나머지는 직접 작성

    @staticmethod
    def get_reward(y_hat, y, n_gram=6): #우리가 피해야되는 for문들의 향연.. for문은 seq이기 때문에 느리다.. 병렬로 짤 필요가 있음.
        import data_loader
        # This method gets the reward based on the sampling result and reference sentence.
        # For now, we uses GLEU in NLTK, but you can used your own well-defined reward function.
        # In addition, GLEU is variation of BLEU, and it is more fit to reinforcement learning.

        # Since we don't calculate reward score exactly as same as multi-bleu.perl,
        # (especialy we do have different tokenization,) I recommend to set n_gram to 6.

        # |y| = (batch_size, length1)
        # |y_hat| = (batch_size, length2)

        scores = []

        # Actually, below is really far from parallized operations.
        # Thus, it may cause slow training.
        for b in range(y.size(0)):
            ref = []
            hyp = []
            for t in range(y.size(1)):
                ref += [str(int(y[b, t]))]
                if y[b, t] == data_loader.EOS:
                    break

            for t in range(y_hat.size(1)):
                hyp += [str(int(y_hat[b, t]))]
                if y_hat[b, t] == data_loader.EOS:
                    break

            # for nltk.bleu & nltk.gleu
            scores += [score_func([ref], hyp, max_len=n_gram) * 100.]

            # for utils.score_sentence
            # scores += [score_func(ref, hyp, 4, smooth = 1)[-1] * 100.]
        scores = torch.FloatTensor(scores).to(y.device)
        # |scores| = (batch_size) 그냥 배치 사이즈를 가짐.

        return scores

    @staticmethod
    def get_gradient(y_hat, y, crit, reward=1):
        from torch.nn import functional as F
        import data_loader
        # |y| = (batch_size, length) # y는 실제 정답X. 내가 이번에 sampling한 녀석.
        # |y_hat| = (batch_size, length, output_size) #softmax 결과값
        # |reward| = (batch_size) #baseline 뺀것
        batch_size = y.size(0)
        output_size = y_hat.size(-1)

        # Before we get the gradient, multiply -reward for each sample and each time-step.
        y_hat = y_hat * -reward.view(-1, 1, 1).expand(*y_hat.size()) #reward.view에서 view는 reshape
                                              # |reward| = (bs, 1, 1) -expand-> (bs, length, |v|(아웃풋 사이즈)) 

        # Generate one-hot to get log-probability.
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = F.one_hot(y.view(-1), num_classes=y_hat.size(-1)).float() #원핫을 먼저 구한다. 만약 4 이면 --> 0 0 0 0 1 0 0 ... 이게 |v|
        # |y| = |y_hat| = (batch_size * length, output_size)
        
        # Generate and apply loss weight to ignore the PAD.
        loss_weight = torch.ones(output_size).to(y.device)
        loss_weight[data_loader.PAD] = 0.
        y = y * loss_weight.view(1, -1)

        log_prob = (y * y_hat).view(batch_size, -1) 
        # |log_prob| = (batch_size, length)
        log_prob.sum().div(batch_size).backward()

        # Like as below, you can also calculate log-probability of sample
        # using Log-Likelihood Loss, which is -NLLLoss.
        # By maximizing NLLLoss, log-probability would be minimized,
        # and it also minimizes "risk", which is "-reward".
        #
        # log_prob = -crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
        #                  y.contiguous().view(-1)
        #                  )
        # log_prob.div(batch_size).backward()

        return log_prob

    @staticmethod
    def step(engine, mini_batch): #수식이 어떻게 구현되었는지 볼것.
        from utils import get_grad_norm, get_parameter_norm

        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()        #트레인 모드로 와라.
        engine.optimizer.zero_grad() #grad 초기화.

        # Raw target variable has both BOS and EOS token. 
        # The output of sequence-to-sequence does not have BOS token. 
        # Thus, remove BOS token for reference.
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        # Take sampling process because set False for is_greedy.
        y_hat, indice = engine.model.search( #새로 생김. 
            x,
            is_greedy=False,
            max_length=engine.config.max_length
        )
        # Based on the result of sampling, get reward.
        actor_reward = MinimumRiskTrainer.get_reward(
            indice,
            y,
            n_gram=engine.config.rl_n_gram #보통 4개까지 하는데, 6개로 정함. 이유는 actor_reward를 돌리게 되는 경우에, 얘는 fully tockenize된 상태이기 때문에 더 많이 봐야된다.
        )
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)
        # |actor_reward| = (batch_size)
        ## << 여기까지 11페이지 수식의 뒤 평균 앞부분

        # Take samples as many as n_samples, and get average rewards for them.
        # I figured out that n_samples = 1 would be enough.
        baseline = [] #미분이 필요가 없다. 
        with torch.no_grad(): #gradient를 안할거니까 no_grad를 해줌.
            for _ in range(engine.config.rl_n_samples): #k번 만큼 for문 돌면서 sampling함. 강사는 1번만함.
                _, sampled_indice = engine.model.search(
                    x,
                    is_greedy=False,
                    max_length=engine.config.max_length,
                )
                baseline += [ #리워드 넣어서 baseline에 추가.
                    MinimumRiskTrainer.get_reward( 
                        sampled_indice,
                        y,
                        n_gram=engine.config.rl_n_gram,
                    )
                ]

            baseline = torch.stack(baseline).sum(dim=0).div(engine.config.rl_n_samples) #여기에 나오는 dim은 없어지는 dimension
            #11페이지 수식의 뒤 평균까지 구했다.
            # |baseline| = (n_samples, batch_size) --> (batch_size)

        # Now, we have relatively expected cumulative reward.
        # Which score can be drawn from actor_reward subtracted by baseline.
        final_reward = actor_reward - baseline
        # |final_reward| = (batch_size)

        # calcuate gradients with back-propagation
        MinimumRiskTrainer.get_gradient(y_hat, indice, engine.crit, reward=final_reward)

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # In orther to avoid gradient exploding, we apply gradient clipping.
        torch_utils.clip_grad_norm_(
            engine.model.parameters(),
            engine.config.max_grad_norm,
        )
        # Take a step of gradient descent.
        engine.optimizer.step()

        return float(actor_reward.mean()), float(baseline.mean()), float(final_reward.mean()), p_norm, g_norm

    @staticmethod
    def validate(engine, mini_batch):
        from utils import get_grad_norm, get_parameter_norm

        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            # feed-forward
            y_hat, indice = engine.model.search(
                x,
                is_greedy=True,
                max_length=engine.config.max_length,
            )
            # |y_hat| = (batch_size, length, output_size)
            # |indice| = (batch_size, length)
            reward = MinimumRiskTrainer.get_reward(
                indice,
                y,
                n_gram=engine.config.rl_n_gram,
            )

        return float(reward.mean())

    @staticmethod
    def attach(trainer, evaluator, verbose=VERBOSE_BATCH_WISE):
        from ignite.engine import Events
        from ignite.metrics import RunningAverage
        from ignite.contrib.handlers.tqdm_logger import ProgressBar

        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'actor')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'baseline')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'reward')
        RunningAverage(output_transform=lambda x: x[3]).attach(trainer, '|param|')
        RunningAverage(output_transform=lambda x: x[4]).attach(trainer, '|g_param|')

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(trainer, ['|param|', '|g_param|', 'actor', 'baseline', 'reward'])

        if verbose >= VERBOSE_EPOCH_WISE:
            @trainer.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_reward = engine.state.metrics['reward']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} BLEU={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_reward,
                ))

        RunningAverage(output_transform=lambda x: x).attach(evaluator, 'BLEU')

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(evaluator, ['BLEU'])

        if verbose >= VERBOSE_EPOCH_WISE:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_bleu = engine.state.metrics['BLEU']
                print('Validation - BLEU={:.2f} best_BLEU={:.2f}'.format(
                    avg_bleu,
                    -engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        from copy import deepcopy

        loss = -float(engine.state.metrics['BLEU'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_bleu = train_engine.state.metrics['actor']
        avg_valid_bleu = engine.state.metrics['BLEU']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % (train_engine.epoch_idx),
                                    '%.2f' % (avg_train_bleu),
                                    '%.2f' % (avg_valid_bleu),
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

