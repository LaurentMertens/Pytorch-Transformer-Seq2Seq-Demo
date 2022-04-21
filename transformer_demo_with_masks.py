"""
Same as transformer_demo.py, except that in this case, the generated samples can have different lenghts,
which introduces the need for masks.
"""
import math

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ############################################################
# Model definition
# ############################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # print(f"pe.shape: {pe.shape}")
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(f"x.shape: {x.shape}")
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class SimpleTransformerModel(nn.Module):
    def __init__(self, nb_tokens: int, emb_size: int, nb_layers=2, nb_heads=4, hid_size=512, dropout=0.25, max_len=30):
        super(SimpleTransformerModel, self).__init__()
        from torch.nn import Transformer
        self.emb_size = emb_size
        self.max_len = max_len

        self.pos_encoder = PositionalEncoding(emb_size, dropout=dropout, max_len=max_len)
        self.embedder = nn.Embedding(nb_tokens, emb_size)

        self.transformer = Transformer(d_model=emb_size, nhead=nb_heads, num_encoder_layers=nb_layers,
                                       num_decoder_layers=nb_layers, dim_feedforward=hid_size, dropout=dropout)

        self.out_lin = nn.Linear(in_features=emb_size, out_features=nb_tokens)

        self.tgt_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).to(device)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def enc_forward(self, src, src_key_padding_mask):
        # Embed source
        src = self.embedder(src) * math.sqrt(self.emb_size)
        # Add positional encoding + reshape into format (seq element, batch element, embedding)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)

        # Push through encoder
        output = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)

        return output

    def dec_forward(self, memory, tgt, tgt_key_padding_mask):
        # Generate target mask, if necessary
        if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.shape[1]:
            mask = self._generate_square_subsequent_mask(tgt.shape[1]).to(device)
            self.tgt_mask = mask

        # Embed target
        tgt = self.embedder(tgt) * math.sqrt(self.emb_size)
        # Add positional encoding + reshape into format (seq element, batch element, embedding)
        tgt = self.pos_encoder(tgt)
        tgt = tgt.transpose(0, 1)

        # Push through decoder + linear output layer
        output = self.out_lin(self.transformer.decoder(memory=memory, tgt=tgt, tgt_mask=self.tgt_mask,
                                                       tgt_key_padding_mask=tgt_key_padding_mask))
        # If using the model to evaluate, also take softmax of final layer to obtain probabilities
        if not self.training:
            output = torch.nn.functional.softmax(output, 2)

        return output

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.enc_forward(src, src_key_padding_mask=src_key_padding_mask)
        output = self.dec_forward(memory, tgt, tgt_key_padding_mask=tgt_key_padding_mask)

        return output

    def greedy_decode(self, src, max_len=None, start_symbol=0, stop_symbol=None):
        """
        Greedy decode input "src": generate output character one at a time, until "stop_symbol" is generated or
        the output exceeds max_len, whichever comes first.

        :param src: input src, 1D tensor
        :param max_len: int
        :param start_symbol: int
        :param stop_symbol: int
        :return: decoded output sequence
        """
        b_training = self.training
        if b_training:
            self.eval()

        if max_len is None:
            max_len = self.max_len
        elif max_len > self.max_len:
            raise ValueError(f"Parameter 'max_len' can not exceed model's own max_len,"
                             f" which is set at {self.max_len}.")
        # Get memory = output from encoder layer
        memory = model.enc_forward(src, src_key_padding_mask=None)
        # Initiate output buffer
        idxs = [start_symbol]
        # Keep track of last predicted symbol
        next_char = start_symbol
        # Keep generating output until "stop_symbol" is generated, or max_len is reached
        while next_char != stop_symbol:
            if len(idxs) == max_len:
                break
            # Convert output buffer to tensor
            ys = torch.LongTensor(idxs).unsqueeze(0).to(device)
            # Push through decoder
            out = self.dec_forward(memory=memory, tgt=ys, tgt_key_padding_mask=None)
            # Get position of max probability of newly predicted character
            # print(f"out.shape: {out.shape}")
            # print(f"out[-1,-1,:]: {out[-1,-1,:]}")
            # print(f"torch.max(out[-1,-1,:], dim=1): {torch.max(out[-1,-1,:], dim=0)}")
            _, next_char = torch.max(out[-1, -1, :], dim=0)
            # print(f"next_char: {next_char}")
            next_char = next_char.item()

            # Append generated character to output buffer
            idxs.append(next_char)

        if b_training:
            self.train()

        return idxs


# ############################################################
# Data generation
# ############################################################
class ToyData(torch.utils.data.Dataset):
    def __init__(self, n_elems=1000, max_length=3, min_val=0, max_val=10, shift=2):
        """
        Generate 'n_elems' source-target pairs, where each source consists of a sequence of 'length'
        consecutive numbers between 'min_val' and 'max_val' (roundabout), and the target is the same sequence
        shifted by 'shift'.

        Generated samples have a random length such that 1 <= length <= max_length.

        :param n_elems: number of elements to generate.
        :param max_length: maximum sequence length.
        :param min_val: minimum value of sequence elements.
        :param max_val: maximum value of sequence elements.
        :param shift: number by which the target sequence will be shifted wrt the source sequence.
        """
        self.max_len = max_length
        self.min_val = min_val
        self.max_val = max_val
        self.sos = '♥'  # 'start-of-sequence' character
        self.eos = '♦'  # 'end-of-sequence' character
        self.pad = '♣'

        self.abc_map = {i: i for i in range(10)}
        for c in [self.sos, self.eos, self.pad]:
            self.abc_map[c] = len(self.abc_map)
        self.cba_map = {v: k for k, v in self.abc_map.items()}

        self.sos_idx = self.abc_map[self.sos]
        self.eos_idx = self.abc_map[self.eos]
        self.pad_idx = self.abc_map[self.pad]

        source = []
        targets = []
        for _ in range(n_elems):
            start = np.random.randint(min_val, max_val)
            length = np.random.randint(1, self.max_len+1)  # max_length - 3
            s = [self.abc_map[(start+i) % max_val] for i in range(length)]\
                + [self.abc_map[self.eos]]
            t = [self.abc_map[self.sos]]\
                + [self.abc_map[(start+shift+i) % max_val] for i in range(length)]\
                + [self.abc_map[self.eos]]
            source.append(s)
            targets.append(t)

        self.source = source
        self.targets = targets

        self.nb_chars = (max_val - min_val) + 3  # '+3' for 'sos', 'eos' and 'pad' characters

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        """
        Retrieves an item for training.
        :param item: Item index
        :return: Return item in the form (x, y). Where x is a vector of features, and y are the ground truth values to
        be predicted.
        """
        s, t = self.source[item], self.targets[item]
        # Pad entries
        if len(s) < self.max_len + 1:
            s += [self.pad_idx] * (self.max_len + 1 - len(s))
        if len(t) < self.max_len + 2:
            t += [self.pad_idx] * (self.max_len + 2 - len(t))

        return torch.Tensor(s).long(), torch.Tensor(t).long()

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        :return: The length of the dataset.
        """
        assert len(self.source) == len(self.targets)
        return len(self.source)

    def idxs_to_str(self, idxs):
        res = ''
        for e in idxs:
            res += str(self.cba_map[e])

        return  res

    def torch_to_str(self, tensor):
        """

        :param tensor: 2D tensor
        :return:
        """
        res = ''
        for row in tensor:
            res += str(self.cba_map[torch.argmax(row).item()])

        return res


# ############################################################
# Train and evaluate model
# ############################################################
if __name__ == '__main__':
    seq_len = 8
    shift = 4
    train_data_loader = torch.utils.data.DataLoader(dataset=ToyData(n_elems=100000, max_length=seq_len, shift=shift),
                                                    batch_size=50)
    test_data = ToyData(n_elems=250, max_length=seq_len, shift=shift)
    ntokens = train_data_loader.dataset.nb_chars  # the size of vocabulary
    emsize = 60  # embedding dimension
    nhid = 256  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    max_len = 30  # max sequence length our model can handle
    model = SimpleTransformerModel(nb_tokens=ntokens, emb_size=emsize, nb_layers=nlayers, nb_heads=nhead,
                                   hid_size=nhid, max_len=max_len, dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_data_loader.dataset.pad_idx)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.25)

    # Define training loop
    def train(nb_epochs=5, print_every=None, padding_idx=0):
        model.train()  # Turn on the train mode
        total_loss = 0.
        for at_epoch in range(nb_epochs):
            for at_sample, (data, target) in enumerate(train_data_loader):
                optimizer.zero_grad()

                data = data.to(device)
                # Non-zero/True positions are ignored
                data_mask = (data == padding_idx)
                tgt = target[:, :-1].to(device)
                tgt_mask = (tgt == padding_idx)
                tgt_y = target[:, 1:].to(device)

                output = model(data, tgt, src_key_padding_mask=data_mask, tgt_key_padding_mask=tgt_mask)
                # print(f"output.shape: {output.shape}")
                output = output.transpose(0, 1)
                # for s in output:
                #     print(train_data_loader.dataset.torch_to_str(s))

                output = output.transpose(1, 2)
                # print(f"output.shape: {output.shape}")

                # data = data[0].to(device)
                # tgt = target[0][:-1].to(device)
                # tgt_y = target[0][1:].to(device)

                # print(f"tgt_y.shape: {tgt_y.shape}")
                loss = criterion(output, tgt_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                total_loss += loss.item()

                if print_every is not None and (at_sample + 1) % print_every == 0:
                    print("-" * 60)
                    print(f"Epoch {at_epoch} :: sample {at_sample + 1}: total_loss = {total_loss:7.5f}")
                    print(f"\tAvg. loss per item: {total_loss / (at_sample + 1):7.5f}")

                    for w in [[x % train_data_loader.dataset.max_val for x in range(1, 1 + 3)]
                              + [train_data_loader.dataset.eos_idx],
                              [x % train_data_loader.dataset.max_val for x in range(5, 5 + 5)]
                              + [train_data_loader.dataset.eos_idx]]:
                        greedy = model.greedy_decode(src=torch.Tensor(w).unsqueeze(0).long().to(device),
                                                     start_symbol=train_data_loader.dataset.sos_idx,
                                                     stop_symbol=train_data_loader.dataset.eos_idx)
                        print(f"{w} --> {train_data_loader.dataset.idxs_to_str(greedy)}")  # Don't print sos
                    eval()

            scheduler.step(epoch=at_epoch)

    # Define evaluation loop
    # We use MSELoss here, because the greedy_decode method does not return probabilities.
    # For this demo, we just want to see the damn thing decrease.
    def eval():
        print("Evaluating...")
        eval_loss = torch.nn.MSELoss()
        total_loss = 0

        for i in range(len(test_data)):
            if (i+1) % 50 == 0:
                print(f"\rAt element {i+1} of {len(test_data)}...", end='', flush=True)
            pred = model.greedy_decode(src=torch.Tensor(test_data.source[i]).unsqueeze(0).long().to(device),
                                       start_symbol=train_data_loader.dataset.sos_idx,
                                       stop_symbol=train_data_loader.dataset.eos_idx)

            # Ignore SOS character
            pred = pred[1:]
            ref = test_data.targets[i][1:]
            # Make of equal length by padding with 0
            if len(ref) > len(pred):
                pred = pred + [0]*(len(ref) - len(pred))
            elif len(pred) > len(ref):
                ref = ref + [0]*(len(pred) - len(ref))

            # Compute MSE loss
            total_loss += eval_loss(torch.Tensor(pred), torch.Tensor(ref))
        print(f"\rAt element {len(test_data)} of {len(test_data)}...")
        print(f"Eval loss: {total_loss:8.3f} :: Avg. loss per element: {total_loss/len(test_data):8.3f}")

    train(print_every=250, padding_idx=train_data_loader.dataset.pad_idx)
