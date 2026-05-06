from layer import *


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, cat_vocab_size=0, cat_emb_dim=8, naics2_vocab_size=0, naics2_emb_dim=8, naics4_vocab_size=0, naics4_emb_dim=8, brand_vocab_size=0, brand_emb_dim=8, parent_vocab_size=0, parent_emb_dim=8):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.seq_length = seq_length
        
        # Embedding layers for static attributes
        self.cat_emb = None
        self.naics2_emb = None
        self.naics4_emb = None
        self.brand_emb = None
        self.parent_emb = None
        
        if cat_vocab_size > 0:
            self.cat_emb = nn.Embedding(cat_vocab_size, cat_emb_dim)
        if naics2_vocab_size > 0:
            self.naics2_emb = nn.Embedding(naics2_vocab_size, naics2_emb_dim)
        if naics4_vocab_size > 0:
            self.naics4_emb = nn.Embedding(naics4_vocab_size, naics4_emb_dim)
        if brand_vocab_size > 0:
            self.brand_emb = nn.Embedding(brand_vocab_size, brand_emb_dim)
        if parent_vocab_size > 0:
            self.parent_emb = nn.Embedding(parent_vocab_size, parent_emb_dim)
        
        # Total embedding dim
        self.num_static_embs = sum([
            cat_vocab_size > 0,
            naics2_vocab_size > 0,
            naics4_vocab_size > 0,
            brand_vocab_size > 0,
            parent_vocab_size > 0
        ])
        
        self.in_dim = in_dim
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None, cat_idx=None, naics2_idx=None, naics4_idx=None, brand_idx=None, parent_idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
            current_seq_len = self.receptive_field
        else:
            current_seq_len = seq_len

        # Build embedding tensor from static attributes
        emb_list = []
        
        if self.cat_emb is not None and cat_idx is not None:
            cat_emb = self.cat_emb(cat_idx)  # (N, cat_emb_dim)
            emb_list.append(cat_emb)
        if self.naics2_emb is not None and naics2_idx is not None:
            naics2_emb = self.naics2_emb(naics2_idx)
            emb_list.append(naics2_emb)
        if self.naics4_emb is not None and naics4_idx is not None:
            naics4_emb = self.naics4_emb(naics4_idx)
            emb_list.append(naics4_emb)
        if self.brand_emb is not None and brand_idx is not None:
            brand_emb = self.brand_emb(brand_idx)
            emb_list.append(brand_emb)
        if self.parent_emb is not None and parent_idx is not None:
            parent_emb = self.parent_emb(parent_idx)
            emb_list.append(parent_emb)
        
        # Concatenate all embeddings: (N, total_emb_dim)
        if len(emb_list) > 0:
            static_emb = torch.cat(emb_list, dim=-1)  # (N, total_emb_dim)
            # Expand to match sequence length: (N, total_emb_dim) -> (1, N, total_emb_dim) -> (1, N, total_emb_dim, T)
            static_emb = static_emb.unsqueeze(0).unsqueeze(-1)  # (1, N, total_emb, 1)
            static_emb = static_emb.expand(-1, -1, -1, current_seq_len)  # (1, N, total_emb, T)
            static_emb = static_emb.transpose(1, 2)  # (1, total_emb, N, T)
            # Pad or project static_emb to match in_dim
            if static_emb.shape[1] < self.in_dim:
                padding = torch.zeros(1, self.in_dim - static_emb.shape[1], static_emb.shape[2], static_emb.shape[3],
                                      device=static_emb.device, dtype=static_emb.dtype)
                static_emb = torch.cat([static_emb, padding], dim=1)
            elif static_emb.shape[1] > self.in_dim:
                static_emb = static_emb[:, :self.in_dim, :, :]
            # Merge with input: (B, in_dim, N, T)
            if input.size(0) != static_emb.size(0):
                static_emb = static_emb.expand(input.size(0), -1, -1, -1)
            x_merged = input + static_emb
        else:
            x_merged = input

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(x_merged)
        skip = self.skip0(F.dropout(x_merged, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
