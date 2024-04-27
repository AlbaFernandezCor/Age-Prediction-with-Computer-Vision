from models.utils_models_dl import resnet34
from torch.utils.data import DataLoader
from HYPERPARAMETERS import BATCH_SIZE
import torch
import math
import tensorflow as tf

GRAYSCALE = True
RANDOM_SEED = 1
NUM_EPOCHS = 5

if tf.config.experimental.list_physical_devices('GPU'):
    DEVICE = '/GPU:0'
else:
    DEVICE = '/CPU:0'

class DeepLearningMethod():
    
    def run(self, train_dataset, test_dataset):
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=1)
        # test_loader = DataLoader(dataset=test_dataset,
        #                           batch_size=BATCH_SIZE,
        #                           shuffle=False,
        #                           num_workers=1)
        model = resnet34(GRAYSCALE)
        self.train_loop(model, train_loader)

    def train_loop(self, model, train_loader):
        best_mae, best_rmse, best_epoch = math.inf, math.inf, -1
        for epoch in range(NUM_EPOCHS):
            model.train()

            for batch_idx, tupla in enumerate(train_loader):
                features = tupla[0]
                targets = tupla[1]

                # features = features.to(DEVICE)
                # targets = targets.to(DEVICE)

                # FORWARD AND BACK PROP
                logits, probas = model(features)
                exit()
                cost = cost_fn(nom_model=LOSS, logits=logits, levels=levels, imp=imp, targets=targets)
                optimizer.zero_grad()
                cost.backward()

                # UPDATE MODEL PARAMETERS
                optimizer.step()

                # LOGGING
                if not batch_idx % 50:
                    s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                        % (epoch + 1, num_epochs, batch_idx,
                            len_train_dataset // BATCH_SIZE, cost))
                    print(s)
                    with open(LOGFILE, 'a') as f:
                        f.write('%s\n' % s)

            model.eval()
            with torch.set_grad_enabled(False):
                valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                                        device=DEVICE, nom_model=LOSS)

            if valid_mae < best_mae:
                best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
                ########## SAVE MODEL #############
                torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))

            s = 'MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d' % (
                valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch)
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

            s = 'Time elapsed: %.2f min' % ((time.time() - start_time) / 60)
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

        model.eval()

