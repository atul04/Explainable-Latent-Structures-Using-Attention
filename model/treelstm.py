import math

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from . import basic


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """

        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl*(fl + 1).sigmoid() + cr*(fr + 1).sigmoid()
             + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class BinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, intra_attention,
                 gumbel_temperature, bidirectional):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.gumbel_temperature = gumbel_temperature
        self.bidirectional = bidirectional
        self.layer = 1
        ################################################################
        #TO be used in Experiments to be done with attention assuming sentence length would not cross 
        #self.att_wt = nn.Parameter(torch.ones(100,hidden_dim)) Used in every experiments
        if(self.bidirectional):
        	self.layer = 2
        if(self.intra_attention):
        	##################################### Individual Attention Weights taken #############################################
        	#self.att_wt = nn.Parameter(torch.rand((200,self.layer*hidden_dim)))
        	#self.att_wt.float()
        	self.att_wt = nn.Linear(in_features=self.layer*hidden_dim,
                                         out_features=1)
        	self.pre_att = nn.Linear(in_features=self.layer*hidden_dim,
                                         out_features=self.layer*hidden_dim)
	################################################################
	
        assert not (self.bidirectional and not self.use_leaf_rnn)

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        if self.bidirectional:
            self.treelstm_layer = BinaryTreeLSTMLayer(2 * hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(2 * hidden_dim))
        else:
            self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal_(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal_(self.leaf_rnn_cell.weight_hh.data)
            init.constant_(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant_(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                init.kaiming_normal_(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal_(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant_(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant_(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal_(self.word_linear.weight.data)
            init.constant_(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        init.normal_(self.comp_query.data, mean=0, std=0.01)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = (self.comp_query * new_h).sum(-1)
        comp_weights = comp_weights / math.sqrt(self.hidden_dim)
        if self.training:
            select_mask = basic.st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature,
                mask=mask)
        else:
            select_mask = basic.greedy_select(logits=comp_weights, mask=mask)
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, input, length, return_select_masks=False):
        max_depth = input.size(1)
        at = None
        #print("Input",input.size())
        #print("Input Size",max_depth)
        #print("Self attention",self.intra_attention)
        length_mask = basic.sequence_mask(sequence_length=length,
                                          max_length=max_depth)
        select_masks = []

        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = input.size()
            zero_state = input.data.new_zeros(batch_size, self.hidden_dim)
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=input[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                length_long = length.long()
                lengths_list = list(length_long.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=input, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = self.word_linear(input)
            state = state.chunk(chunks=2, dim=2)
        nodes = []
        embd  = []
        #print("State ",state[0].shape)
        if self.intra_attention:
            #print("State Size",state)
            a = state[0].chunk(chunks=state[0].shape[1],dim=1)
            #print("H initial ",a[0].shape)
            #print("H initial ",a[0][0])
            c = []
            e = []
            for i in a:
            	c.append(F.relu(self.pre_att(i)))
            	e.append(i)
            #print("After operation H initial",c[0].shape)
            #print("After operation H initial",c[0][0])
            #exit()
            ##a = [ ele.squeeze(1) for ele in a ]
            #print("State[0] Size",a[0].shape)
            embd.extend(c)
            nodes.extend(e)
        for i in range(max_depth - 1):
            h, c = state
            l = (h[:, :-1, :], c[:, :-1, :])
            r = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(l=l, r=r)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i+1:])
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                if self.intra_attention:
                    embd.append(F.relu(self.pre_att(selected_h.unsqueeze(1))))
                    nodes.append(selected_h.unsqueeze(1))
                    #print("Selected H",nodes[-1].size())
            done_mask = length_mask[:, i+1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if self.intra_attention and i >= max_depth - 2:
                embd.append(F.relu(self.pre_att(state[0])))
                nodes.append(state[0])
        h, c = state
        #print("h",h[0][0])
        if self.intra_attention:
            ########################################################### This code segment is returning a good attention scores############
            ##print("Length Mask",length_mask.shape)
            ##print("Length Mask [0]",length_mask[0])
            #att_mask = length_mask[:, 1:].clone()# grtting a score of 0.7170 WIthout attention 0.725
            ##att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)# getting a score of 0.6620
            ##print("Attention Mask Size",att_mask.size())
            #att_mask = att_mask.float()
            ## nodes: (batch_size, num_tree_nodes, hidden_dim)
            ##for node in nodes:
            ##    print(node.shape)
            #nodes = torch.cat(nodes, dim=1)
            ##print("Nodes : ",nodes.shape)
            ##print("Node [0][0]",nodes[0][0])
            #att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes)
            ##print("Attention Mask Expand Size",att_mask_expand.size())
            ##print("Attention Mask Expand [0][0]",att_mask_expand[0][0])
            #nodes = nodes * att_mask_expand
            ##print("Node [0][0]",nodes[0][0])
            ## nodes_mean: (batch_size, hidden_dim, 1)
            #nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            ## att_weights: (batch_size, num_tree_nodes)
            #att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            #att_weights = basic.masked_softmax(
            #    logits=att_weights, mask=att_mask)
            ## att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            #att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            ## h: (batch_size, 1, 2 * hidden_dim)
            #h = (att_weights_expand * nodes).sum(1)
            #h = h.unsqueeze(1)
            ############################################################### Uptill here the attention scores on the basis of mean value
            
            ############################################################### Writing my own attention routine ######################
            #print("Length Mask",length_mask.shape)
            #print("Att_wt ",self.att_wt.shape)
            #self.att_wt[torch.isnan(self.att_wt)] = 0
            #self.att_wt.float()
            get_cuda_device = length_mask.device
            #print("Device ",get_cuda_device)
            batch = length_mask.shape[0]
            #print("batch",batch)
            # Whole
            # For individual attention
            #For single attention we dont need this #pad_len = self.att_wt.shape[0] - 2*length_mask.shape[1] +1 # (because in length_mask we are excluding 1 element)
            #non leaf
            #pad_len = self.att_wt.shape[0] - length_mask.shape[1] +1 
            #print("pad_len",pad_len)
            dtype = torch.FloatTensor
            #For single attention we dont need this #att_pad = torch.zeros([batch,pad_len]).to(get_cuda_device)
            #print("attention pad",att_pad.shape)
            # whole
            #For single attention we dont need this#att_mask = torch.cat([length_mask.type(dtype),length_mask[:, 1:].type(dtype),att_pad.type(dtype)], dim=1)
            #For single attention 
            att_mask = torch.cat([length_mask.type(dtype),length_mask[:, 1:].type(dtype)], dim=1)
            # non leaf
            #att_mask = torch.cat([length_mask[:, 1:].type(dtype),att_pad.type(dtype)], dim=1)
            #print("Attention Mask Size",att_mask[0])
            att_mask = att_mask.float().to(get_cuda_device)
            #print("Attention Mask Size",att_mask)
            # nodes: (batch_size, num_tree_nodes, hidden_dim)
            nodes = torch.cat(nodes, dim=1)
            #print("Nodes : ",nodes.shape)
            #For single attention we dont need this #nodes_pad = torch.zeros([batch,pad_len,self.layer*self.hidden_dim]).to(get_cuda_device)
            #print("Nodes pad ",nodes_pad.shape)
            #For single attention we dont need this #nodes = torch.cat([nodes, nodes_pad.float()],dim=1)
            #print("Nodes : ",nodes.shape)
            ## Embedding Acquisition EMbd
            # embd: (batch_size, num_tree_nodes, hidden_dim)
            embd = torch.cat(embd, dim=1)
            #print("embd : ",embd.shape)
            #For single attention we dont need this #embd_pad = torch.zeros([batch,pad_len,self.layer*self.hidden_dim]).to(get_cuda_device)
            #print("Embd pad ",embd_pad.shape)
            #For single attention we dont need this #embd = torch.cat([embd, embd_pad.float()],dim=1)
            #print("Embd : ",embd.shape)
            att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes).to(get_cuda_device)
            #print("Attention Mask/snli/SaveModel/Gumble_Intra_att_Linear_embd_non_leaf_log_leaf_rnn_batch_norm_test_840B_0.1_bidirectional_32_Single/model-5.70-0.4508-0.8505.pkl Expand Size",att_mask_expand.size())
            
            nodes = nodes.to(get_cuda_device)
            nodes = nodes * att_mask_expand
            nodes[torch.isnan(nodes)] = 0
            nodes.float()
            
            embd = embd.to(get_cuda_device)
            embd = embd * att_mask_expand
            embd[torch.isnan(embd)] = 0
            embd.float()
            
            
            ##############    Individual Attention Weights taken    ############################ 
            
            ##a_t = torch.tanh(self.att_wt.t().to(get_cuda_device))
            #a_t = self.att_wt.t().to(get_cuda_device)
            ## att_weights=att_scores: (batch_size, num_tree_nodes)
            #att_scores = torch.matmul(embd,a_t)
            ##print("Attention scores[0]",att_scores[0])
            ##att_scores = torch.tanh(att_scores)
            ##print("Attention scores",att_scores.shape)
            ##print("Attention scores[0]",att_scores[0])
            ##print("Attention scores [4]",att_scores[4])
            ##print("Attention scores [45]",att_scores[45])
            #att_scores = torch.diagonal(att_scores, offset=0, dim1=-2, dim2=-1)
            ##print("Attention scores",att_scores.shape)
            ##att_scores[torch.isnan(att_scores)] = 0
            #att_scores.float()
            ##print("After Attention scores [0]",att_scores[0])
            ##print("After Attention scores [4]",att_scores[4])
            ##print("AfterAttention scores [45]",att_scores[45])
            
            ################################       Uptill here    ##################################
            
            ############################# Single Attention vector   ################################
            att_scores = self.att_wt(embd).squeeze(2)
            #########################################################################################
            #print("Attention scores",att_scores[0])       
            att_weights = basic.masked_softmax(
                logits=att_scores, mask=att_mask)
            #print("att_n weights",att_weights)
            #exit()
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            at = att_weights
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1)
            h = h.unsqueeze(1)
            #print("h_att",h[0][0])
            #exit()
            ################################################################ Uptill here own attention routine######################
        assert h.size(1) == 1 and c.size(1) == 1
	
        if not return_select_masks:    
            return h.squeeze(1), c.squeeze(1), at
        else:
            return h.squeeze(1), c.squeeze(1), select_masks, at 
