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

        self.num_cat_embs = sum([
            cat_vocab_size > 0,
            naics2_vocab_size > 0,
            naics4_vocab_size > 0,
            brand_vocab_size > 0,
            parent_vocab_size > 0
        ])
        self.total_cat_emb_dim = (
            (cat_vocab_size > 0) * cat_emb_dim +
            (naics2_vocab_size > 0) * naics2_emb_dim +
            (naics4_vocab_size > 0) * naics4_emb_dim +
            (brand_vocab_size > 0) * brand_emb_dim +
            (parent_vocab_size > 0) * parent_emb_dim
        )

        self.in_dim = in_dim
        self.residual_channels = residual_channels
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

        num_nodes = input.size(2)

        D_s = max(0, self.in_dim - 1 - self.num_cat_embs)

        emb_list = []
        emb_dims = []
        for name, emb, provided_idx in [
            ("cat", self.cat_emb, cat_idx),
            ("naics2", self.naics2_emb, naics2_idx),
            ("naics4", self.naics4_emb, naics4_idx),
            ("brand", self.brand_emb, brand_idx),
            ("parent", self.parent_emb, parent_idx),
        ]:
            if emb is None:
                continue
            if idx is None and provided_idx is not None:
                e = emb(provided_idx)
            else:
                cat_channel = 1 + D_s + len(emb_list)
                e = emb(input[:, cat_channel, :, 0].long())
            emb_list.append(e)
            emb_dims.append(e.size(-1))

        if len(emb_list) > 0:
            static_emb = torch.cat(emb_list, dim=-1)
            B, N, total_emb = static_emb.shape
            static_emb = static_emb.view(B, total_emb, N, 1)
            static_emb = static_emb.expand(B, total_emb, N, current_seq_len).contiguous()
            if static_emb.shape[1] < self.in_dim:
                pad = torch.zeros(B, self.in_dim - static_emb.shape[1], N, current_seq_len,
                                  device=static_emb.device, dtype=static_emb.dtype)
                static_emb = torch.cat([static_emb, pad], dim=1)
            elif static_emb.shape[1] > self.in_dim:
                static_emb = static_emb[:, :self.in_dim, :, :]
            x = input + static_emb
            skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        else:
            x = input
            skip = self.skip0(F.dropout(x, self.dropout, training=self.training))

        x = self.start_conv(x)

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                if idx is not None:
                    adp = self.predefined_A[idx][:, idx]
                else:
                    adp = self.predefined_A

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
