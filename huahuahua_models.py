import torch.nn as nn
import torch
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, classification=False):
        super(LSTM, self).__init__()
        self.hidden_size = output_size
        self.cell_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Linear(input_size + output_size, output_size)
        self.classification = classification
        if self.classification:
            self.output_dense = nn.Linear(output_size, 2)

    def forward(self, input, h_t, c_t):
        combined = torch.cat((input, h_t), 1)

        i = self.sigmoid(self.gate(combined))
        f = self.sigmoid(self.gate(combined))
        c = torch.add(torch.mul(c_t, f), torch.mul(self.tanh(self.gate(combined)), i))
        o = self.sigmoid(self.gate(combined))

        h = torch.mul(self.tanh(c), o)

        if self.classification:
            output = self.output_dense(h)
        else:
            output = h

        return output, h, c

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.cell_size))


class RLSTM(nn.Module):
    def __init__(self, input_size, output_size, media_size, classification=False):
        super(RLSTM, self).__init__()
        self.hidden_size = output_size
        self.cell_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.media_gate = nn.Linear(media_size, output_size)
        self.input_gate = nn.Linear(input_size + output_size, output_size)
        self.classification = classification
        if self.classification:
            self.output_dense = nn.Linear(output_size, 2)

    def forward(self, input, media, h_t, c_t):
        combined = torch.cat((input, h_t), 1)

        i = self.sigmoid(self.input_gate(combined))
        f = self.sigmoid(self.input_gate(combined))
        c = torch.add(torch.mul(c_t, f), torch.mul(self.tanh(self.input_gate(combined)), i))
        o = self.sigmoid(self.input_gate(combined))

        c_r = self.tanh(c)
        c_f = torch.add(c, torch.add(-c_r, torch.mul(c_r, self.sigmoid(self.media_gate(media)))))

        h = torch.mul(self.tanh(c_f), o)

        if self.classification:
            output = self.output_dense(h)
        else:
            output = h

        return output, h, c

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.cell_size))


class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_channels, filters, kernel_size, classification=False):
        super(ConvLSTM, self).__init__()
        self.hidden_size = input_size
        self.cell_size = input_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.input_conv = nn.Conv2d(in_channels=input_channels, out_channels=filters,
                                    kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        self.hidden_conv = nn.Conv2d(in_channels=filters, out_channels=filters,
                                     kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        self.classification = classification
        if self.classification:
            self.flatten = nn.Flatten()
            self.output_dense = nn.Linear(input_size[0] * input_size[1] * filters, 2)

    def forward(self, input, h_t, c_t):
        x_i = self.input_conv(input)
        x_f = self.input_conv(input)
        x_c = self.input_conv(input)
        x_o = self.input_conv(input)

        h_i = self.hidden_conv(h_t)
        h_f = self.hidden_conv(h_t)
        h_c = self.hidden_conv(h_t)
        h_o = self.hidden_conv(h_t)

        i = self.sigmoid(torch.add(x_i, h_i))
        f = self.sigmoid(torch.add(x_f, h_f))
        c = torch.add(torch.mul(f, c_t), torch.mul(i, self.tanh(torch.add(x_c, h_c))))
        o = self.sigmoid(torch.add(x_o, h_o))
        h = torch.mul(self.tanh(c), o)

        if self.classification:
            flatten = self.flatten(h)
            output = self.output_dense(flatten)
        else:
            output = h

        return output, h, c

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.cell_size))


class RConvLSTM(nn.Module):
    def __init__(self, input_size, input_channels, filters, kernel_size, media_channels, classification=False):
        super(RConvLSTM, self).__init__()
        self.hidden_size = input_size
        self.cell_size = input_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.input_conv = nn.Conv2d(in_channels=input_channels, out_channels=filters,
                                    kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        self.hidden_conv = nn.Conv2d(in_channels=filters, out_channels=filters,
                                     kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        self.media_conv = nn.Conv2d(in_channels=media_channels, out_channels=filters,
                                    kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        self.classification = classification
        if self.classification:
            self.flatten = nn.Flatten()
            self.output_dense = nn.Linear(input_size[0] * input_size[1] * filters, 2)

    def forward(self, input, media, h_t, c_t):
        x_i = self.input_conv(input)
        x_f = self.input_conv(input)
        x_c = self.input_conv(input)
        x_o = self.input_conv(input)

        h_i = self.hidden_conv(h_t)
        h_f = self.hidden_conv(h_t)
        h_c = self.hidden_conv(h_t)
        h_o = self.hidden_conv(h_t)

        i = self.sigmoid(torch.add(x_i, h_i))
        f = self.sigmoid(torch.add(x_f, h_f))
        c = torch.add(torch.mul(f, c_t), torch.mul(i, self.tanh(torch.add(x_c, h_c))))
        o = self.sigmoid(torch.add(x_o, h_o))

        c_r = self.tanh(c)
        c_f = torch.add(c, torch.add(-c_r, torch.mul(c_r, self.sigmoid(self.media_conv(media)))))

        h = torch.mul(self.tanh(c_f), o)

        if self.classification:
            flatten = self.flatten(h)
            output = self.output_dense(flatten)
        else:
            output = h

        return output, h, c

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.cell_size))
