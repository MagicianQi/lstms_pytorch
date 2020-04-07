# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np

from huahuahua_models import RLSTM

# -----------------------Training parameters-----------------------

num_epochs = 10
GPU_id = "cuda:0"   # Specifies the graphics card
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
lr_init = 0.01      # Initial learning rate
weights_path = "./pkl/"     # Model save path
weights_save_interval = 5   # Model save interval

# -----------------------Mock training data-----------------------

inputs = [[torch.randn(1, 10) for _ in range(5)] for _ in range(100)]   # train data, dimension：[batch,timeStamp,vector]
medias = [[torch.randn(1, 6) for _ in range(5)] for _ in range(100)]    # media data, structure as above
labels = [torch.from_numpy(np.array([random.randint(0, 1)])) for _ in range(100)]       # label in（0, 1）

# -----------------------Model initialization-----------------------

model = RLSTM(input_size=10, output_size=32, media_size=6, classification=True)
model = model.to(device)

# -----------------------Loss function、optimizer、learning rate decay-----------------------

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# -----------------------Train-----------------------

model.train(mode=True)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 20)

    exp_lr_scheduler.step()

    iteration = 0
    for input, media, label in zip(inputs, medias, labels):
        iteration += 1
        optimizer.zero_grad()

        label = label.to(device)
        out = None
        hidden = model.init_hidden()
        cell = model.init_cell()
        hidden = hidden.to(device)
        cell = cell.to(device)
        for i, m in zip(input, media):
            i = i.to(device)
            m = m.to(device)
            out, hidden, cell = model(i, m, hidden, cell)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        print("Epoch: {}\tIter: {}\tLoss: {}".format(epoch, iteration, loss.item()))

    if epoch % weights_save_interval == 0:
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        torch.save(model.state_dict(), weights_path + "epoch_{}.pkl".format(epoch))

# -----------------------Inference-----------------------

inputs_val = [[torch.randn(1, 10) for _ in range(5)] for _ in range(10)]  # test data
medias_val = [[torch.randn(1, 6) for _ in range(5)] for _ in range(10)]

model.eval()

for input_val, media_val in zip(inputs_val, medias_val):
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        hidden = model.init_hidden()
        cell = model.init_cell()
        hidden = hidden.to(device)
        cell = cell.to(device)
        out = None
        for i, m in zip(input_val, media_val):
            i = i.to(device)
            m = m.to(device)
            out, hidden, cell = model(i, m, hidden, cell)
        _, out_val = torch.max(out, 1)
        print(out_val.cpu().numpy())
