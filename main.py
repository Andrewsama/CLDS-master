import time
from os.path import join

import torch

import Procedure
import register
import utils
import world
from register import dataset

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
torch.autograd.set_detect_anomaly(True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
d = world.device
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

best0_ndcg, best0_recall, best0_pre = 0, 0, 0
best1_ndcg, best1_recall, best1_pre = 0, 0, 0
best0_ndcg_cold, best0_recall_cold, best0_pre_cold = 0, 0, 0
best1_ndcg_cold, best1_recall_cold, best1_pre_cold = 0, 0, 0
low_count, low_count_cold = 0, 0
start = time.time()

tip = 'pre'
try:
    for epoch in range(world.TRAIN_epochs + 1):
        print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')

        if epoch > 2000 and (epoch % 10 == 1 or epoch == world.TRAIN_epochs):
            print("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, False)
            # results_cold = Procedure.Test(dataset, Recmodel, epoch, True)
            if results['ndcg'][0] < best0_ndcg:
                low_count += 1
                if low_count == 30:
                    if epoch > 1000:
                        break
                    else:
                        low_count = 0
            else:
                best0_recall = results['recall'][0]
                best0_ndcg = results['ndcg'][0]
                best0_pre = results['precision'][0]
                low_count = 0

            if results['ndcg'][1] >= best1_ndcg:
                best1_recall = results['recall'][1]
                best1_ndcg = results['ndcg'][1]
                best1_pre = results['precision'][1]
        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
        print(f'[saved][BPR aver loss{loss:.3e}]')
    end = time.time()
    print('The total time:',(end - start) / 60)
        # torch.save(Recmodel.state_dict(), weight_file)
finally:
    print(f"best precision at 10:{best0_pre}")
    print(f"best precision at 20:{best1_pre}")
    print(f"best recall at 10:{best0_recall}")
    print(f"best recall at 20:{best1_recall}")
    print(f"best ndcg at 10:{best0_ndcg}")
    print(f"best ndcg at 20:{best1_ndcg}")
