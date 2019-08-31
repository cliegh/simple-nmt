import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_loader
from simple_nmt.search import SingleBeamSearchSpace


class Attention(nn.Module): # 어텐션 : 키-벨류 펑션. back propogation이 가능한.
                            # 쿼리를 linear transformation을 통해서 만들었다.
                            
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False) #linear 레이어. 내가 찾고싶은 쿼리를 만드는 과정. 
                                                                    # 내 상태가 주어졌을때 알맞은 쿼리가 리턴값으로 나옴. bias는 true 줘도 똑같이 나옴.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |h_t_tgt| = (batch_size, 1, hidden_size)
        # |mask| = (batch_size, length)
                                    #(bs, 1, hs)
                                               #(bs, hs) << 스퀴즈 이후
        query = self.linear(h_t_tgt.squeeze(1)).unsqueeze(-1) # w*X 가 됨. (bs, hs, 1) << 언스퀴즈를 통해 1을 이쪽으로 옮겨줌
        # |query| = (batch_size, hidden_size, 1)

        weight = torch.bmm(h_src, query).squeeze(-1) #정제되지 않고 임의의 유사도에 대한 크기만 들어있음. 이후 소프트맥스 씌우면 0~1사이로 감
        # |weight| = (batch_size, length) # (인코더의 각 샘플별, 인코더의 타임스탭별) 가중치
        if mask is not None: 
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0, masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch, the weight for empty time-step would be set to 0.
            weight.masked_fill_(mask, -float('inf')) #1이 있는 위치에 -무한대를 해주어라. 쓸데없는 곳에 있는 weight 들은 다 날림.
        weight = self.softmax(weight) #원하는 위치에 대한 것만 가져오고 0~1로 구성함

                                                    #(bs, 1, l)
        context_vector = torch.bmm(weight.unsqueeze(1), h_src) # (bs, l, hs) ==> bmm ==> (bs, 1, hs)
        # |context_vector| = (batch_size, 1, hidden_size)

        return context_vector


class Encoder(nn.Module): #문장을 받아서 압축해서 하나의 벡터(하나의 점)로 만듬. 비슷한 문장, 비슷한 벡터

    def __init__(self, word_vec_dim, hidden_size, n_layers=4, dropout_p=.2): #RNN은 hidden_size를 갖는다, dropout : LSTM 에서
        super(Encoder, self).__init__()

        # Be aware of value of 'batch_first' parameter.
        # Also, its hidden_size is half of original hidden_size, because it is bidirectional.
        self.rnn = nn.LSTM(word_vec_dim,
                           int(hidden_size / 2), #bi-direction에 쓸거라 나누기 2
                           num_layers=n_layers,
                           dropout=dropout_p,
                           bidirectional=True,
                           batch_first=True # (bs , l , ws)에서 bs , l 의 위치가 바뀜
                           )

    def forward(self, emb): # 임베딩한 텐서가 들어온다
        # |emb| = (batch_size, length, word_vec_dim)

        if isinstance(emb, tuple):#RNN에 쓸때 packed sequence 
            x, lengths = emb # 임베딩 벡터가 실제 
            x = pack(x, lengths.tolist(), batch_first=True)# x를 length에 대해서 패킹

            # Below is how pack_padded_sequence works.
            # As you can see, PackedSequence object has information about mini-batch-wise information, not time-step-wise information.
            # 
            # a = [torch.tensor([1,2,3]), torch.tensor([3,4])] 
            # b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
            # >>>>
            # tensor([[ 1,  2,  3],
            #     [ 3,  4,  0]])
            # torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2]
            # >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))  <- packed sequence의 역할.
            # 명시적으로 패딩을 먹일 수 있다. 마지막 0이 패드다. 라고 알려주는것
            # RNN은 한 타임스태프씩 받기 때문에 1,3,2,4,3 으로 들어옴. 그리고 2개 2개 하나씩 묶어서. (1,3)(2,4)(3)
        else:
            x = emb

        y, h = self.rnn(x) # pytorch documentation 참고하면 있음. lstm의 입력과 출력에 대한 정의에서, PPT page9 참조. 
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2) 
        # num_layers *2 는 bi-direction 의미, 각 층의 방향별로 샘플별 히든 스테이트가 나옴

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h


class Decoder(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        # Be aware of value of 'batch_first' parameter and 'bidirectional' parameter.
        self.rnn = nn.LSTM(word_vec_dim + hidden_size,
                           hidden_size,
                           num_layers=n_layers,
                           dropout=dropout_p,
                           bidirectional=False,
                           batch_first=True
                           )

    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| = (batch_size, 1, word_vec_dim)
        # |h_t_1_tilde| = (batch_size, 1, hidden_size)
        # |h_t_1[0]| = (n_layers, batch_size, hidden_size)
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1) #h_t_1[0] 은 hidden state

        if h_t_1_tilde is None:
            # If this is the first time-step,
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        # Input feeding trick.
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1) #(bs, 1, wv+hs)

        # Unlike encoder, decoder must take an input for sequentially.
        y, h = self.rnn(x, h_t_1) # y : (bs, 1, hs) 시간이 존재, h : (#layers, bs, hs) 시간이 존재하지 않는다. 

        return y, h


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size) #vocabulary 사이즈로 빼주는 linear
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length, hidden_size)

        y = self.softmax(self.output(x))
        # |y| = (batch_size, length, output_size)

        # Return log-probability instead of just probability.
        return y


class Seq2Seq(nn.Module):

    def __init__(self,
                 input_size,
                 word_vec_dim,
                 hidden_size,
                 output_size,
                 n_layers=4,
                 dropout_p=.2
                 ):
        self.input_size = input_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_dim) #각각 임베딩 레이어 선언. 서로 다른 vocabulary 사이즈기 때문에 따로 선언
        self.emb_dec = nn.Embedding(output_size, word_vec_dim)
        # 임베딩 벡터의 동작 원리 : word 2 vec을 넣어주어야 하는거아니냐? X. 원핫벡터를 넣어서 자동으로 해준다.

        # 인코더, 디코더에 넣어줌
        self.encoder = Encoder(word_vec_dim, 
                               hidden_size,
                               n_layers=n_layers,
                               dropout_p=dropout_p
                               )
        self.decoder = Decoder(word_vec_dim,
                               hidden_size,
                               n_layers=n_layers,
                               dropout_p=dropout_p
                               )
        self.attn = Attention(hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x, length): #마스크 생성.
        mask = []

        max_length = max(length)
        for l in length: #길이만큼 0을 채우고, 그 뒤에 남는 부분을 1을 채우고 하는 식으로. for문으로 어쩔수 없이 구성됨.
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples, 
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, (max_length - l))
                                    ], dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples, 
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).byte()

        return mask

    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []

        hiddens, cells = encoder_hiddens #튜플. cells -> (#layers*2 , bs, hs/2)

        # i-th and (i+1)-th layer is opposite direction.
        # Also, each direction of layer is half hidden size.
        # Therefore, we concatenate both directions to 1 hidden size layer.
        for i in range(0, hiddens.size(0), 2): #각 층별로. hidden의 0 은 두칸씩 간다. for문은 지양해야한다. 우리는 병렬로 해야하기 때문
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim=-1)]
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim=-1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)

        return (new_hiddens, new_cells)

    def forward(self, src, tgt): #소스하고 타겟 둘다 받아야함. 훈련할때는 티처포싱이기 때문.
        batch_size = tgt.size(0)

        mask = None #마스크란? 책에 있음. 어텐션을 통해 유사도를 비교해야하는데, 들어가면 안되는 garbege값이 들어갈 수 있으니, 
                    #패딩을 주기위해 의도적으로 1로 mask를 해 주어서 결과적으로 attention weight가 0이 되도록 해줌.
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        # Get word embedding vectors for every time-step of input sentence.
        emb_src = self.emb_src(x) #원핫벡터가 임베딩 레이어 통과.
        # |emb_src| = (batch_size, length, word_vec_dim)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2)

        # Merge bidirectional to uni-directional
        # We need to convert size from (n_layers * 2, batch_size, hidden_size / 2) to (n_layers, batch_size, hidden_size).
        # Thus, the converting operation will not working with just 'view' method.
        h_0_tgt, c_0_tgt = h_0_tgt
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size, #h_0_tgt -> 0 : (#l*2, bs, hs/2) , 1 : (bs,#l*2,hs/2)
                                                            -1,
                                                            self.hidden_size # (bs, #layers , hs) transpose 한 다음에 view를 해주어야 한다.
                                                            ).transpose(0, 1).contiguous() # (#layers, bs, hs)
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous() #바이디렉션을 유니디렉션으로 바꿔주면서 모양을 맞춰주는 작업.
        # You can use 'merge_encoder_hiddens' method, instead of using above 3 lines.
        # 'merge_encoder_hiddens' method works with non-parallel way.
        # h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        h_0_tgt = (h_0_tgt, c_0_tgt)

        emb_tgt = self.emb_dec(tgt)
        # |emb_tgt| = (batch_size, length, word_vec_dim)
        h_tilde = []

        h_t_tilde = None
        decoder_hidden = h_0_tgt
        # Run decoder until the end of the time-step.

        #디코딩의 시작
        for t in range(tgt.size(1)): # 어쩔 수 없는 포문. input feeding 때문에. 실제로 이곳이 병목 발생 장소.
            # Teacher Forcing: take each input from training set, not from the last time-step's output.
            # Because of Teacher Forcing, training procedure and inference procedure becomes different.
            # Of course, because of sequential running in decoder, this causes severe bottle-neck.
            emb_t = emb_tgt[:, t, :].unsqueeze(1) # 한 타임스텝만 띄어울꺼다. unsqueeze 내가 원하는 위치에 1을 넣는다.
            # |emb_t| = (batch_size, 1, word_vec_dim) # 배치사이즈, 렝스, 워드벡터디멘젼 : 하나만 떼오면 가운데 1이 없어짐.
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                          h_t_tilde,
                                                          decoder_hidden
                                                          )
            # |decoder_output| = (batch_size, 1, hidden_size)
            # |decoder_hidden| = (n_layers, batch_size, hidden_size)

            context_vector = self.attn(h_src, decoder_output, mask) # context_vector 은 attention 클래스에서 나옴.
            # |context_vector| = (batch_size, 1, hidden_size)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector 
                                                         ], dim=-1))) # 여기까지 (bs, 1, 2*hs) // concat은 위에 정의되어있음
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            h_tilde += [h_t_tilde] #모든 타임스탭의 h_tilde를 합침. 문장이 끝날때까지 돈다.

        h_tilde = torch.cat(h_tilde, dim=1) #
        # |h_tilde| = (batch_size, length, hidden_size)

        y_hat = self.generator(h_tilde)
        # |y_hat| = (batch_size, length, output_size)

        return y_hat

    def search(self, src, is_greedy=True, max_length=255): #샘플링 또는 그리디 수행을 위한 서치. 빔서치는 아래 따로 구현
        mask, x_length = None, None

        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        h_0_tgt, c_0_tgt = h_0_tgt
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        h_0_tgt = (h_0_tgt, c_0_tgt)

        # Fill a vector, which has 'batch_size' dimension, with BOS value.
        y = x.new(batch_size, 1).zero_() + data_loader.BOS
        is_undone = x.new_ones(batch_size, 1).float()
        decoder_hidden = h_0_tgt
        h_t_tilde, y_hats, indice = None, [], []
        
        # Repeat a loop while sum of 'is_undone' flag is bigger than 0, or current time-step is smaller than maximum length.
        while is_undone.sum() > 0 and len(indice) < max_length:
            # Unlike training procedure, take the last time-step's output during the inference.
            emb_t = self.emb_dec(y)
            # |emb_t| = (batch_size, 1, word_vec_dim)

            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                          h_t_tilde,
                                                          decoder_hidden
                                                          )
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector
                                                         ], dim=-1)))
            y_hat = self.generator(h_t_tilde)
            # |y_hat| = (batch_size, 1, output_size)
            y_hats += [y_hat]

            if is_greedy:
                y = torch.topk(y_hat, 1, dim=-1)[1].squeeze(-1)
            else:
                # Take a random sampling based on the multinoulli distribution.
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)
            # Put PAD if the sample is done.
            y = y.masked_fill_((1. - is_undone).byte(), data_loader.PAD)
            is_undone = is_undone * torch.ne(y, data_loader.EOS).float()
            # |y| = (batch_size, 1)
            # |is_undone| = (batch_size, 1)
            indice += [y]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=-1)
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        return y_hats, indice

    def batch_beam_search(self, src, beam_size=5, max_length=255, n_best=1): # 빔서치는 따로 구현.
        mask, x_length = None, None

        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        h_0_tgt, c_0_tgt = h_0_tgt
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        h_0_tgt = (h_0_tgt, c_0_tgt) ## forward와 여기까지는 동일

        # initialize 'SingleBeamSearchSpace' as many as batch_size
        spaces = [SingleBeamSearchSpace((h_0_tgt[0][:, i, :].unsqueeze(1), #각 샘플별 히든스테이트, 셀 스테이트에 넣어줌.
                                         h_0_tgt[1][:, i, :].unsqueeze(1) 
                                         ),
                                        None,
                                        beam_size,
                                        max_length=max_length
                                        ) for i in range(batch_size)]# 각 문장별로 singlebeamsearchspace를 만든것.
        done_cnt = [space.is_done() for space in spaces]

        length = 0
        # Run loop while sum of 'done_cnt' is smaller than batch_size, or length is still smaller than max_length.
        while sum(done_cnt) < batch_size and length <= max_length:
            # current_batch_size = sum(done_cnt) * beam_size

            # Initialize fabricated variables.
            # As far as batch-beam-search is running, 
            # temporary batch-size for fabricated mini-batch is 'beam_size'-times bigger than original batch_size.
            fab_input, fab_hidden, fab_cell, fab_h_t_tilde = [], [], [], [] #가짜 배치들
            fab_h_src, fab_mask = [], []
            
            # Build fabricated mini-batch in non-parallel way.
            # This may cause a bottle-neck.
            for i, space in enumerate(spaces):
                if space.is_done() == 0:  # Batchfy only if the inference for the sample is still not finished. 문장이 끝나지 않았으면
                    y_hat_, (hidden_, cell_), h_t_tilde_ = space.get_batch()

                    fab_input += [y_hat_]
                    fab_hidden += [hidden_]
                    fab_cell += [cell_]
                    if h_t_tilde_ is not None:
                        fab_h_t_tilde += [h_t_tilde_]
                    else:
                        fab_h_t_tilde = None

                    fab_h_src += [h_src[i, :, :]] * beam_size
                    fab_mask += [mask[i, :]] * beam_size

            # Now, concatenate list of tensors. 가짜 미치배치들을 concat? 한다.
            fab_input = torch.cat(fab_input, dim=0)
            fab_hidden = torch.cat(fab_hidden, dim=1)
            fab_cell = torch.cat(fab_cell, dim=1)
            if fab_h_t_tilde is not None:
                fab_h_t_tilde = torch.cat(fab_h_t_tilde, dim=0)
            fab_h_src = torch.stack(fab_h_src)
            fab_mask = torch.stack(fab_mask)
            # current_batch_size = sum(done_cnt) * beam_size
            # |fab_input| = (current_batch_size, 1)
            # |fab_hidden| = (n_layers, current_batch_size, hidden_size)
            # |fab_cell| = (n_layers, current_batch_size, hidden_size)
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
            # |fab_h_src| = (current_batch_size, length, hidden_size)
            # |fab_mask| = (current_batch_size, length)

            emb_t = self.emb_dec(fab_input)
            # |emb_t| = (current_batch_size, 1, word_vec_dim)

            fab_decoder_output, (fab_hidden, fab_cell) = self.decoder(emb_t,
                                                                      fab_h_t_tilde,
                                                                      (fab_hidden, fab_cell)
                                                                      )
            # |fab_decoder_output| = (current_batch_size, 1, hidden_size)
            context_vector = self.attn(fab_h_src, fab_decoder_output, fab_mask)
            # |context_vector| = (current_batch_size, 1, hidden_size)
            fab_h_t_tilde = self.tanh(self.concat(torch.cat([fab_decoder_output,
                                                             context_vector
                                                             ], dim=-1)))
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
            y_hat = self.generator(fab_h_t_tilde)
            # |y_hat| = (current_batch_size, 1, output_size)

            # separate the result for each sample.
            cnt = 0
            for space in spaces: #ppt 그림에서 붙어있는 translated Fabricated Mini-bach를 쪼개줌.
                if space.is_done() == 0:
                    # Decide a range of each sample.
                    from_index = cnt * beam_size
                    to_index = from_index + beam_size

                    # pick k-best results for each sample. 
                    space.collect_result(y_hat[from_index:to_index],
                                         (fab_hidden[:, from_index:to_index, :],
                                          fab_cell[:, from_index:to_index, :]
                                          ),
                                         fab_h_t_tilde[from_index:to_index]
                                         )
                    cnt += 1

            done_cnt = [space.is_done() for space in spaces]
            length += 1

        # pick n-best hypothesis.
        batch_sentences = []
        batch_probs = []

        # Collect the results.
        for i, space in enumerate(spaces):
            sentences, probs = space.get_n_best(n_best)

            batch_sentences += [sentences]
            batch_probs += [probs]

        return batch_sentences, batch_probs
