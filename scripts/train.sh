#!/usr/bin/env bash

python -m enet.run.ee.runner --earlystop 10 --restart 999999 --optimizer "adadelta" --lr 1 --webd "" --batch 32 --epochs 99999 --device "cuda:0" --out "models/enet-081" --hps "{'wemb_dim': 200, 'wemb_ft': True, 'wemb_dp': 0.5, 'pemb_dim': 50, 'pemb_dp': 0.5, 'eemb_dim': 50, 'eemb_dp': 0.5, 'psemb_dim': 50, 'psemb_dp': 0.5, 'lstm_dim': 220, 'lstm_layers': 1, 'lstm_dp': 0, 'gcn_et': 3, 'gcn_use_bn': True, 'gcn_layers': 3, 'gcn_dp': 0.5, 'sa_dim': 300, 'use_highway': True, 'loss_alpha': 5}" >& enet081.log &
