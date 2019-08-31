from operator import itemgetter

import torch
import torch.nn as nn

import data_loader

LENGTH_PENALTY = 1.2
MIN_LENGTH = 5


class SingleBeamSearchSpace():

    def __init__(self,
                 hidden,
                 h_t_tilde=None,
                 beam_size=5,
                 max_length=255
                 ):
        self.beam_size = beam_size
        self.max_length = max_length

        super(SingleBeamSearchSpace, self).__init__()

        # To put data to same device.
        self.device = hidden[0].device #디바이스 저장. 같은 디바이스 동작 위함.
        # Inferred word index for each time-step. For now, initialized with initial time-step.
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS] # 뽑힌 워드를 저장하는 히스토리 저장용. 맨 처음이니 다 BOS를 넣음.
        # Index origin of current beam.
        self.prev_beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1] # 이전에 어디서 뽑혀왔냐. 빔사이즈 개수만큼 단어가 뽑힘. 제로로 채워 넣어놓음.
        # Cumulative log-probability for each beam.
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)] # 누적확률. 0 인것 말고 나머지는 -무한대로 해서 없앰. 로그확률이니 확률값은 0은 1, -무한대는 0이 됨.
        # 1 if it is done else 0
        self.masks = [torch.ByteTensor(beam_size).zero_().to(self.device)] # 마스크.

        # We don't need to remember every time-step of hidden states: prev_hidden, prev_cell, prev_h_t_tilde
        # What we need is remember just last one.
        # Future work: make this class to deal with any necessary information for other architecture, such as Transformer.

        # |hidden[0]| = (n_layers, 1, hidden_size)
        self.prev_hidden = torch.cat([hidden[0]] * beam_size, dim=1)
        self.prev_cell = torch.cat([hidden[1]] * beam_size, dim=1)
        # |prev_hidden| = (n_layers, beam_size, hidden_size)
        # |prev_cell| = (n_layers, beam_size, hidden_size)

        # |h_t_tilde| = (batch_size = 1, 1, hidden_size)
        self.prev_h_t_tilde = torch.cat([h_t_tilde] * beam_size,
                                        dim=0
                                        ) if h_t_tilde is not None else None
        # |prev_h_t_tilde| = (beam_size, 1, hidden_size)

        self.current_time_step = 0
        self.done_cnt = 0

    def get_length_penalty(self,
                           length,
                           alpha=LENGTH_PENALTY, # 맨 위에 정의되어 있음. 하이퍼파라미터인데. 굳이 튜닝 필요 없고 많이 쓰는 값 쓰면 됨.
                           min_length=MIN_LENGTH
                           ):
        # Calculate length-penalty, because shorter sentence usually have bigger probability.
        # Thus, we need to put penalty for shorter one.
        p = (1 + length) ** alpha / (1 + min_length) ** alpha

        return p

    def is_done(self):
        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0

    def get_batch(self): # PPT에서 get batch 하는것
        y_hat = self.word_indice[-1].unsqueeze(-1) # 마지막 워드 인다이스에 -1 언스퀴즈 줌.
        hidden = (self.prev_hidden, self.prev_cell)
        h_t_tilde = self.prev_h_t_tilde

        # |y_hat| = (beam_size, 1) #이렇게 나옴
        # |hidden| = (n_layers, beam_size, hidden_size)
        # |h_t_tilde| = (beam_size, 1, hidden_size) or None
        return y_hat, hidden, h_t_tilde

    def collect_result(self, y_hat, hidden, h_t_tilde): # 각 빔서치 스페이스에 대해서, 누적확률 계산하고 베스트 뽑고 하는 작업을 함.
        # |y_hat| = (beam_size, 1, output_size) # 각 빔별로 들어오는것이 아니라, 
        # |hidden| = (n_layers, beam_size, hidden_size)
        # |h_t_tilde| = (beam_size, 1, hidden_size) #빔사이즈 * 히든사이즈
        # 배치사이즈가 빔사이즈로 다 교체되었다. 병렬로 계산하는 것이란 의미
        output_size = y_hat.size(-1)

        self.current_time_step += 1 #현재 타임스탭 증가

        # Calculate cumulative log-probability.
        # First, fill -inf value to last cumulative probability, if the beam is already finished.
        # Second, expand -inf filled cumulative probability to fit to 'y_hat'. (beam_size) --> (beam_size, 1, 1) --> (beam_size, 1, output_size)
        # cumulative_probs 가 빔사이즈만큼 있다. expand를 써서 output_size를 만듬
        # Third, add expanded cumulative probability to 'y_hat'
        #누적확률 구함
                                                                        #끝났으면 1, 안끝났으면 0
                                                            #이전 타임스탭까지 누적확률
                        #현재 타임스탭까지 누적확률
        cumulative_prob = y_hat + self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf')).view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        # Now, we have new top log-probability and its index. We picked top index as many as 'beam_size'.
        # Be aware that we picked top-k from whole batch through 'view(-1)'.
        #k개의 top을 뽑아주는 함수. 실제 best인 값과 위치를 tuple로 반환.
        top_log_prob, top_indice = torch.topk(cumulative_prob.view(-1), #(beam_size, 1, |v|) -> (bs * |v|)
                                              self.beam_size,
                                              dim=-1
                                              )
        # |top_log_prob| = (beam_size)
        # |top_indice| = (beam_size)

        # Because we picked from whole batch, original word index should be calculated again.
        self.word_indice += [top_indice.fmod(output_size)]
        # Also, we can get an index of beam, which has top-k log-probability search result.
        self.prev_beam_indice += [top_indice.div(output_size).long()]

        # Add results to history boards.
        self.cumulative_probs += [top_log_prob]
        self.masks += [torch.eq(self.word_indice[-1],
                       data_loader.EOS)
                       ]  # Set finish mask if we got EOS.
        # Calculate a number of finished beams.
        self.done_cnt += self.masks[-1].float().sum() #지금까지 EOS가 몇개 나왔나 누적해서 가져옴.

        # Set hidden states for next time-step, using 'index_select' method.
        self.prev_hidden = torch.index_select(hidden[0], #(#layers, bs, hs)
                                              dim=1,
                                              index=self.prev_beam_indice[-1] #내가 원하는 빔 인다이스를 뽑아서 내가 원하는 형태로 만들어줌
                                              ).contiguous()
        self.prev_cell = torch.index_select(hidden[1],
                                            dim=1,
                                            index=self.prev_beam_indice[-1]
                                            ).contiguous()
        self.prev_h_t_tilde = torch.index_select(h_t_tilde,
                                                 dim=0,
                                                 index=self.prev_beam_indice[-1]
                                                 ).contiguous()

    def get_n_best(self, n=1):
        sentences, probs, founds = [], [], []

        for t in range(len(self.word_indice)):  # for each time-step,
            for b in range(self.beam_size):  # for each beam,
                if self.masks[t][b] == 1:  # if we had EOS on this time-step and beam,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] / self.get_length_penalty(t)]
                    founds += [(t, b)]

        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'):
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b]]
                    founds += [(t, b)]

        # Sort and take n-best.
        sorted_founds_with_probs = sorted(zip(founds, probs),
                                          key=itemgetter(1),
                                          reverse=True
                                          )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.prev_beam_indice[t][b]

            sentences += [sentence]
            probs += [prob]

        return sentences, probs
