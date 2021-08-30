import torch
import torch.nn as nn
import math
from torch.nn import Parameter

import collections


def _gen_mask(shape, drop_prob):
    """Generate a droppout mask."""
    keep_prob = 1. - drop_prob
    #mask = tf.random_uniform(shape, dtype=tf.float32)
    mask = torch.FloatTensor(shape[0], shape[1]).uniform_(0, 1)
    mask = torch.floor(mask + keep_prob)
    return mask

class OPT_LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, device=torch.device('cuda')):
        super(OPT_LSTMCell, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activations = [torch.relu, torch.tanh, torch.sigmoid]
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        # Variational Dropout
        #self.h_mask = _gen_mask((4 * hidden_size, hidden_size), 0.5).to(self.device)
        self.init_weight()

    def init_weight(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        :return:
        """
        nn.init.orthogonal_(self.weight_hh.data)
        nn.init.xavier_uniform_(self.weight_ih.data)
        # nn.init.zeros_(self.bias_hh.data)
        # nn.init.zeros_(self.bias_ih.data)
        self.bias_hh.data.fill_(0)
        self.bias_ih.data.fill_(0)
        self.bias_hh.data[self.hidden_size:2 * self.hidden_size] = 1
        self.bias_ih.data[self.hidden_size:2 * self.hidden_size] = 1


    def forward(self, input, conf, state, h_mask, train=True):
        hx, cx = state
        # Weight drop
        # if train:
        #     self.weight_hh.data = self.weight_hh.data * h_mask
        # self.weight_ih.data = self.weight_ih.data * self.h_mask
        # self.weight_hh.data = self.weight_hh.data * self.h_mask
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        f_1 = self.activations[conf[1]]
        f_2 = self.activations[conf[2]]
        ingate = f_1(ingate)
        forgetgate = f_1(forgetgate)
        outgate = f_1(outgate)
        cellgate = f_2(cellgate)

        cy = cx * forgetgate + ingate * cellgate
        hy = outgate * f_2(cy)

        return (hy, cy)


class Searchable_DKT_COMB(nn.Module):
    """Deep Knowledge tracing model"""

    def __init__(self, num_skills, num_steps, conf, device=torch.device('cuda'), hidden_size=100):
        super(Searchable_DKT_COMB, self).__init__()

        self.device = device
        self.conf = conf
        self.num_layers = 10
        self.embedding_size = hidden_size

        self.lstm1 = nn.LSTMCell(2, hidden_size)
        # Apply Variational Dropout for inputs and outputs
        # Use weight dropout for hidden to hidden ?
        # self.inp_mask = _gen_mask((1, hidden_size), 0.2).to(self.device)
        # self.out_mask = _gen_mask((1, hidden_size), 0.2).to(self.device)

        self.num_steps = num_steps

        # self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        # self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        # self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        # self.lstm4 = nn.LSTMCell(hidden_size, hidden_size)


        self.lstm1 = OPT_LSTMCell(hidden_size, hidden_size, device)
        self.lstm2 = OPT_LSTMCell(hidden_size, hidden_size, device)
        self.lstm3 = OPT_LSTMCell(hidden_size, hidden_size, device)
        self.lstm4 = OPT_LSTMCell(hidden_size, hidden_size, device)

        self.lstms = [self.lstm1, self.lstm2, self.lstm3, self.lstm4]
        self.tag_embedding = nn.Embedding(num_skills, self.embedding_size)
        #self.tag_embedding.weight.requires_grad = False

        self.m1 = nn.Linear(2, hidden_size)
        self.m2 = nn.Linear(1, hidden_size)
        self.m3 = nn.Linear(1, hidden_size)
        self.m4 = nn.Linear(1, hidden_size)
        self.decoder = nn.Linear(hidden_size * 2, num_skills)
        self.hh_connect = nn.Linear(hidden_size * 2, hidden_size)
        self.encoders = []
        for i in range(num_skills):
            self.encoders.append(nn.Linear(100, hidden_size).to(device))

        # self.fusion_layers = collections.defaultdict(dict)
        # for idx in range(self.num_layers):
        #     for jdx in range(idx + 1, self.num_layers):
        #         self.fusion_layers[idx][jdx] = nn.Linear(200, 100).to(device)


        self.activations = [torch.relu, torch.tanh, torch.sigmoid]
        #
        # self.w_z = collections.defaultdict(dict)
        # self.w_r = collections.defaultdict(dict)
        # self.w_w = collections.defaultdict(dict)
        #
        # for idx in range(self.num_layers):
        #     for jdx in range(idx + 1, self.num_layers):
        #         if idx < 4 and jdx < 4:
        #             self.w_z[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)
        #             self.w_r[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)
        #             self.w_w[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)
        #         elif idx < 4 and jdx >=4:
        #             self.w_z[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)
        #             self.w_r[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)
        #             self.w_w[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)
        #         else:
        #             self.w_z[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)
        #             self.w_r[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)
        #             self.w_w[idx][jdx] = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(self.device)


        # nas_v0
        self.w_c_x = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w_c_h = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w_h_x = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w_h_h = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w_h_s = nn.Linear(hidden_size * 2, hidden_size).to(self.device)

        self.w_c = collections.defaultdict(dict)
        self.w_h = collections.defaultdict(dict)
        for idx in range(self.num_layers):
            for jdx in range(idx + 1, self.num_layers):
                self.w_c[idx][jdx] = nn.Linear(hidden_size, hidden_size).to(self.device)
                self.w_h[idx][jdx] = nn.Linear(hidden_size, hidden_size).to(self.device)




        # self.input_drop = VariationalDropout(0.3,
        #                                      batch_first=True)
        # self.output_drop = VariationalDropout(0.3,
        #                                       batch_first=True)


        self.init_weights()

        self.rnn_type = 'LSTM'
        self.nhid = hidden_size
        #self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.num_skills = num_skills
        self.num_steps = num_steps

    def init_weights(self):
        # pass
        nn.init.orthogonal_(self.w_c_x.weight.data)
        nn.init.orthogonal_(self.w_c_h.weight.data)
        nn.init.orthogonal_(self.w_h_x.weight.data)
        nn.init.orthogonal_(self.w_h_h.weight.data)
        for idx in range(self.num_layers):
            for jdx in range(idx + 1, self.num_layers):
                nn.init.orthogonal_(self.w_c[idx][jdx].weight.data)
                nn.init.orthogonal_(self.w_h[idx][jdx].weight.data)


    def _nas_cell(self, x, hidden, h_mask, train):
        """Multi-layer LSMT
        """
        m1, m2, m3, m4 = x
        out_q = [m1, m2, m3, m4]
        #h_1, h_2, c_1, c_2 = hidden
        h, c = hidden
        #in_h = [h_1, h_2]
        #in_c = [c_1, c_2]
        out_h = []
        out_c = []
        for i, l in enumerate(self.conf):
            #input = self.input_drop(out_q[l[0]])
            #input = out_q[l[0]] * self.inp_mask
            input = out_q[l[0]]
            h, c = self.lstms[i](input, l, (h, c), h_mask, train)
            #out_h.append(h)
            #out_c.append(c)
        return (h, c)


    def _nas_cell_v0(self, x, h_prev, h_mask, train):
        """Multi-layer
        """
        m1, m2, m3, m4 = x
        out_q = [m1, m2, m3, m4]
        process_q = []
        h = h_prev

        for m in out_q:
            c_1 = torch.sigmoid(self.w_c_x(m) + self.w_c_h(h))
            h_1 = c_1 * torch.tanh(self.w_h_x(m) + self.w_h_h(h)) + (1 - c_1) * h
            process_q.append(h_1)

        # l in the format of [prev_node, activation_function]
        c_node = 4
        leaf_nodes = list(range(4, 4 + len(self.conf)))
        for l in self.conf:
            prev_out = process_q[l[0]]
            if l[0] in leaf_nodes:
                leaf_nodes.remove(l[0])
            f = self.activations[l[1]]
            c_t = torch.sigmoid(self.w_c[l[0]][c_node](prev_out))
            h_t = c_t * f(self.w_h[l[0]][c_node](prev_out)) + (1 - c_t) * prev_out
            process_q.append(h_t)
            c_node += 1

        # # Concatenate leaf nodes for final output
        accu_hidden = process_q[leaf_nodes[0]]
        for i in range(1, len(leaf_nodes)):
            accu_hidden += process_q[leaf_nodes[i]]
        next_s = accu_hidden / len(leaf_nodes)

        #next_s = self.w_h_s(torch.cat((process_q[4], process_q[5]), dim=1))

        return next_s


    def forward(self, input, hidden, routers_info=None, train=True):
        #input_1, input_2, input_3 = input
        input_1, input_2, input_3, input_4 = input
        outputs = []
        #h_1, c_1, h_2, c_2, h_3, c_3  = hidden
        h_t = hidden
        h_mask = _gen_mask((4 * self.hidden_size, self.hidden_size), 0.1).to(self.device)

        h_queue = []
        for i in range(self.num_steps):
            # modality 1
            m1_data = input_1[: ,i]
            m2_data = input_2[:, i]
            m3_data = input_3[:, i]
            m4_data = input_4[:, i]

            current_skills = routers_info[:, i]
            tag_embedding = self.tag_embedding(current_skills)
            #input_data = self.embedding_m1(step_data)

            #### Use torch.bnn

            # Create (b,1,m)
            # Step 1, embedding
            out_1 = self.m1(m1_data)
            out_2 = self.m2(m2_data)
            out_3 = self.m3(m3_data)
            out_4 = self.m4(m4_data)

            out_embeddings = [out_1, out_2, out_3, out_4]
            #h_t, c = self._nas_cell(out_embeddings, (h_t, c), h_mask, train)
            h_t = self._nas_cell_v0(out_embeddings, h_t, h_mask, train)

            # if len(h_queue) <= 50:
            #     h_queue.append(h_t)
            # else:
            #     h_queue.append(h_t)
            #     h_queue.pop(0)

            # weighted_h = self.hh_connect(torch.cat((h_queue[0], h_t), dim=1))
            # masked_h = weighted_h * self.out_mask
            #masked_h = h_t * self.out_mask
            #masked_h = torch.cat((h_1_t, h_2_t), dim=1)
            masked_h = h_t
            tagged_h = torch.cat((masked_h, tag_embedding), dim=1)
            tmp_data = self.decoder(tagged_h)
            outputs += [tmp_data]


        outputs = torch.stack(outputs, 1).squeeze(2)
        output = outputs.contiguous().view(outputs.size(0) * outputs.size(1), outputs.size(2))

        # decoded = self.decoder(outputs.contiguous().view(outputs.size(0) * outputs.size(1), outputs.size(2)))
        return output, h_t


    def init_hidden(self, bsz):
        h_1_t = torch.zeros(bsz, self.nhid, dtype=torch.float)
        #h_2_t = torch.zeros(bsz, self.nhid, dtype=torch.float)
        c_1_t = torch.zeros(bsz, self.nhid, dtype=torch.float)
        #c_2_t = torch.zeros(bsz, self.nhid, dtype=torch.float)
        return h_1_t