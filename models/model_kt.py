import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepKnowledgeTracing(nn.Module):
    """Deep Knowledge tracing model"""

    def __init__(self, hidden_size, num_skills, num_steps, dropout=0.5):
        super(DeepKnowledgeTracing, self).__init__()

        self.device = torch.device('cuda')
        #
        # embedding_size1 = 100
        # embedding_size2 = 50
        # self.embedding_m1 = nn.Embedding(num_skills * 2, embedding_size1)
        # self.embedding_m2 = nn.Embedding(num_skills * 5, embedding_size2)
        # self.embedding_m3 = nn.Embedding(num_skills * 3, embedding_size2)


        #self.attn = Attention(hidden_size, embedding_size)
        #self.drop = nn.Dropout(dropout)
        #self.encoder = nn.Embedding(ntoken, ninp)

        # if rnn_type in ['LSTM', 'GRU']:
        #     self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
        # else:
        #     try:
        #         nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        #     except KeyError:
        #         raise ValueError("""An invalid option for `--model` was supplied,
        #                              options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        #     self.rnn = nn.RNN(embedding_size, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.lstm1 = nn.LSTMCell(2, hidden_size)
        # self.lstm2 = nn.LSTMCell(num_skills * 5, hidden_size)
        # self.lstm3 = nn.LSTMCell(num_skills * 3, hidden_size)

        self.num_steps = num_steps

        # self.decoder = nn.Linear(hidden_size * 3, num_skills)
        # self.decoder_s = nn.Linear(hidden_size , num_skills)

        # self.lstm1 = CustomLSTMCell(input_size, hidden_size)
        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        #self.lstm2 = CustomLSTMCell(hidden_size, hidden_size)

        self.m1 = nn.Linear(2, 10)
        self.m2 = nn.Linear(1, 10)
        self.decoder = nn.Linear(hidden_size, num_skills)
        self.encoders = []
        for i in range(num_skills):
            self.encoders.append(nn.Linear(20, hidden_size))


        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     pass
            # if hidden_size != input_size:
            #     raise ValueError('When using the tied flag, nhid must be equal to emsize')
            # self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = 'LSTM'
        self.nhid = hidden_size
        #self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.num_skills = num_skills
        self.num_steps = num_steps

    def init_weights(self):
        initrange = 0.05
        # #self.encoder.weight.data.uniform_(-initrange, initrange)
        # for s in self.skills:
        #     s.bias.data.zero_()
        #     s.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, routers_info=None):
        #input_1, input_2, input_3 = input
        input_1, input_2 = input
        outputs = []
        #h_1, c_1, h_2, c_2, h_3, c_3  = hidden
        h_t, c_t = hidden

        for i in range(self.num_steps):
            # modality 1
            input_data = input_1[:,i]
            m2_data = input_2[:, i]

            current_skills = routers_info[:, i]
            #input_data = self.embedding_m1(step_data)

            # ######## Use for loop slow
            # # Routing
            # batch_size = 32
            # # current_pred = []
            # tmp_data = torch.FloatTensor(batch_size, self.hidden_size)
            # tmp_data.zero_()
            # for b in range(batch_size):
            #
            #     out_1 = self.m1(input_data[b])
            #     out_2 = self.m2(m2_data[b])
            #
            #     fused = torch.cat((out_1, out_2), 0)
            #
            #     out = self.encoders[current_skills[b]](fused)
            #     # out = self.routers[0](output[b])
            #     # current_pred.append(out)
            #     tmp_data[b] = out
            ######### Use for loop, slow

            #### Use torch.bnn

            # Create (b,1,m)
            out_1 = self.m1(input_data)
            out_2 = self.m2(m2_data)
            fused = torch.cat((out_1, out_2), dim=1)
            fused = torch.unsqueeze(fused, 1)

            # Create (b,m,n), DO WE ALSO NEED TO ADD BIAS?
            ws = [torch.transpose(l.weight.data, 0, 1) for l in [self.encoders[current_skills[b]] for b in range(32)]]
            bs = [l.bias.data for l in [self.encoders[current_skills[b]] for b in range(32)]]
            b_W = torch.stack(ws)
            b_B = torch.stack(bs)
            tmp_data = torch.bmm(fused, b_W).squeeze()
            tmp_data = tmp_data + b_B


            input_data = tmp_data

            h_t, c_t = self.lstm1(input_data, (h_t, c_t))
            # h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # output = self.decoder(h_t2)
            # output = self.decoder(h_t)
            output = h_t

            tmp_data = self.decoder(output)
            outputs += [tmp_data]

            # Add drop out
            #h_t = self.drop(h_t)
            #h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            # modality 2
            # step_data = input_2[:, i]
            # #input_data = self.embedding_m2(step_data)
            # input_data = step_data
            # h_2, c_2 = self.lstm2(input_data, (h_2, c_2))
            #
            # # modality 3
            # step_data = input_3[:, i]
            # #input_data = self.embedding_m3(step_data)
            # input_data = step_data
            # h_3, c_3 = self.lstm3(input_data, (h_3, c_3))

            # fusion
            #output = self.decoder(torch.cat((h_1, h_2, h_3), dim=1))
            #output = self.decoder_s(h_1)

        outputs = torch.stack(outputs, 1).squeeze(2)
        output = outputs.contiguous().view(outputs.size(0) * outputs.size(1), outputs.size(2))

        # decoded = self.decoder(outputs.contiguous().view(outputs.size(0) * outputs.size(1), outputs.size(2)))
        return output, (h_t, c_t)

    # def init_hidden(self, bsz):
    #     h_1 = torch.zeros(bsz, self.nhid, dtype=torch.float)
    #     c_1 = torch.zeros(bsz, self.nhid, dtype=torch.float)
    #
    #     h_2 = torch.zeros(bsz, self.nhid, dtype=torch.float)
    #     c_2 = torch.zeros(bsz, self.nhid, dtype=torch.float)
    #
    #     h_3 = torch.zeros(bsz, self.nhid, dtype=torch.float)
    #     c_3 = torch.zeros(bsz, self.nhid, dtype=torch.float)
    #
    #     return (h_1, c_1, h_2, c_2, h_3, c_3)

    def init_hidden(self, bsz):
        h_t = torch.zeros(bsz, self.nhid, dtype=torch.float)
        c_t = torch.zeros(bsz, self.nhid, dtype=torch.float)
        return (h_t, c_t)
