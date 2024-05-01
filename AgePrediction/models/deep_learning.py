from models.utils_models_dl import resnet34
from torch.utils.data import DataLoader
from HYPERPARAMETERS import BATCH_SIZE
import torch
import torch.nn as nn
import torch.optim as optim
import math
import tensorflow as tf
import os

GRAYSCALE = False
RANDOM_SEED = 1
NUM_EPOCHS = 5

if tf.config.experimental.list_physical_devices('GPU'):
    DEVICE = '/GPU:0'
else:
    DEVICE = '/CPU:0'


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0., 0., 0
    for i, (features, targets) in enumerate(data_loader):
        # features = features.to(device)
        # targets = targets.to(device)

        outs = model(features)
        _, predicted_labels = torch.max(outs, 1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets) ** 2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse

class DeepLearningMethod():
    
    def run(self, train_dataset, test_dataset, valid_dataset):

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=1)
        test_loader = DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=1)
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=1)
        model = resnet34(GRAYSCALE)
        self.train_loop(model, train_loader, test_loader, valid_loader)

    def train_loop(self, model, train_loader, test_loader, valid_loader):
        best_mae, best_rmse, best_epoch = math.inf, math.inf, -1
        # Definir la función de pérdida
        criterion = nn.MSELoss()

        # Definir el optimizador
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for batch_idx, tupla in enumerate(train_loader):
                print(f"Batch: {batch_idx}")
                features = tupla[0]
                targets = tupla[1]
                # features = features.to(DEVICE)
                # targets = targets.to(DEVICE)

                # FORWARD AND BACK PROP
                outputs = model(features)
                # Calcular la pérdida
                loss = criterion(outputs.squeeze(), targets.float())  # Squeeze elimina las dimensiones de tamaño 1
                                                                    # y labels.float() convierte las etiquetas a float
                
                # Backward pass y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * features.size(0)  # Multiplicamos por el tamaño del lote para tener una pérdida acumulativa
            
            # Calcular la pérdida promedio por epoca
            epoch_loss = running_loss / train_loader.__len__()
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')

            model.eval()
            with torch.set_grad_enabled(False):  # save memory during inference

                train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                        device=DEVICE)
                valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                                        device=DEVICE)
                test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                        device=DEVICE)

                s = 'MAE/RMSE: | Train: %.2f/%.2f | Valid: %.2f/%.2f | Test: %.2f/%.2f' % (
                    train_mae, torch.sqrt(train_mse),
                    valid_mae, torch.sqrt(valid_mse),
                    test_mae, torch.sqrt(test_mse))
                print(s)

