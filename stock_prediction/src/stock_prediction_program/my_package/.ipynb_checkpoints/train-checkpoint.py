import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from .model import Stacked_VAE, BiLSTM
#from .model import Stacked_VAE, BiLSTM

def train_vae(train_loader, test_loader, feats_train, symbol_id, n_hidden, n_latent, n_layers, learning_rate, n_epoch, is_train=True):
    
    model = Stacked_VAE(n_in = feats_train.shape[1], n_hidden = n_hidden, n_latent = n_latent, n_layers = n_layers)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_function = nn.MSELoss()

    save_dir = 'my_path/vae'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if is_train:
        training_loss = []
        validation_loss = []
        best_val_loss = float('inf')
        best_epoch = -1
        for epoch in range(n_epoch):
            epoch_loss = 0
            epoch_val_loss = 0
    
            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                outputs, weight_loss = model(data)
                loss = loss_function(outputs, data)
                loss = loss + torch.mean(weight_loss)
                #recon_batch, mu, log_var = model(data)
                #loss = vae_loss(recon_batch, data, mu, log_var)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
    
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    outputs, weight_loss = model(data)
                    loss = loss_function(outputs, data)
                    loss = loss + torch.mean(weight_loss)
                    #recon_batch, mu, log_var = model(data)
                    #loss = vae_loss(recon_batch, data, mu, log_var)
                    epoch_val_loss += loss.item()
    
            epoch_loss /= len(train_loader)
            epoch_val_loss /= len(test_loader)
    
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                
                # Stacked AutoEncoder 학습파일 저장
                best_checkpoint_path = f'my_path/vae/{symbol_id}_vae.pt'
                
                try:
                    best_checkpoint_path = os.path.join(save_dir, f'{symbol_id}_vae.pt')
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'validation_loss': epoch_val_loss,
                    }, best_checkpoint_path)
                except Exception as e:
                    print(f'Symbol: {symbol_id} Error: {str(e)}')
            
            training_loss.append(epoch_loss)
            validation_loss.append(epoch_val_loss)
    
            if epoch % 50 == 0:
                print('Epoch {}, Training loss {:.4f}, Validation loss {:.4f}'.format(epoch, epoch_loss, epoch_val_loss))

    # Stacked AutoEncoder 학습파일 불러오기
    else:
        try:
            best_checkpoint_path = os.path.join(save_dir, f'{symbol_id}_vae.pt')
            checkpoint = torch.load(best_checkpoint_path)

            #best_checkpoint_path = f'my_path/vae/{symbol_id}_vae.pt'
            #checkpoint = torch.load(os.path.join(os.path.dirname(__file__), best_checkpoint_path))
        
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_loss = checkpoint['loss']
        except Exception as e:
            print(f'Symbol: {symbol_id} Error: {str(e)}')

    return model

def train_bilstm(train_loader, symbol_id, input_size, hidden_size, n_layers, dropout, output_size, num_epochs, learning_rate, device, len_sequence=True):
    model = BiLSTM(input_size, hidden_size, n_layers, dropout, output_size, device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    save_dir = 'my_path/bilstm'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if len_sequence:
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.unsqueeze(2)
                targets = targets.unsqueeze(2)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1)%10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}')

    else:
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            if (epoch+1)%10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}')

    # BiLSTM 학습파일 저장
    try:
        #file_path = f'my_path/bilstm/{symbol_id}_bilstm.pt'
        file_path = os.path.join(save_dir, f'{symbol_id}_bilstm.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
        }, file_path)
        
    except Exception as e:
        print(f'Symbol: {symbol_id} Error: {str(e)}')

    return model

def bilstm_inference(sequences, symbol_id, input_size, hidden_size, n_layers, dropout, output_size, window_size, device, len_sequence=True):
    model = BiLSTM(input_size, hidden_size, n_layers, dropout, output_size, device)
    model.to(device)

    save_dir = 'my_path/bilstm'

    # BiLSTM 학습파일 불러오기
    try:
        file_path = os.path.join(save_dir, f'{symbol_id}_bilstm.pt')
        checkpoint = torch.load(file_path)

        #file_path = f'my_path/bilstm/{symbol_id}_bilstm.pt'
        #checkpoint = torch.load(os.path.join(os.path.dirname(__file__), file_path))
        
        model.load_state_dict(checkpoint['model_state_dict'])
            
    except Exception as e:
        print(f'Symbol: {symbol_id} Error: {str(e)}')

    if len_sequence:
        model.eval()
        with torch.no_grad():
            last_sequences = sequences[-window_size: ]
            last_sequences = last_sequences.to(device)
            train_outputs = model(last_sequences.unsqueeze(2))
    else:
        model.eval()
        with torch.no_grad():
            last_sequences = sequences[-window_size: ]
            last_sequences = last_sequences.to(device)
            train_outputs = model(last_sequences)

    return train_outputs