# -*- coding: UTF-8 -*-​ 
import torch,copy,os
from torch import nn,utils
from smn_model import SMNModel
from data_processor import DataProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(512)

class Config:
    def __init__(self):
        self.data_path = {
            "train": "../drive_data/MyDrive/dataset/dialogue/ubuntu_train_subtask_1.json",
            "dev": "../drive_data/MyDrive/dataset/dialogue/ubuntu_dev_subtask_1.json",
            "test": "../drive_data/MyDrive/dataset/dialogue/ubuntu_test_subtask_1.json"
        }
        self.vocab_path = "../drive_data/MyDrive/dataset/dialogue/vocab.txt"
        self.model_save_path = "../drive_data/MyDrive/dataset/stm_model_param.pkl"
        self.update_vocab = True

        self.vocab_size = 50000
        self.embed_dim = 200
        self.hidden_size = 50
        self.out_channels = 8
        self.fusion_method = "last"
        
        self.max_turn_num = 10
        self.max_seq_len = 50
        self.candidates_set_size = 2 #Rn@k: n=2，10，100, k=1

        self.batch_size = 12
        self.epochs = 50
        self.dropout = 0.2
        self.lr = 0.0002
        self.num_classes = self.candidates_set_size

        self.device = device

def eval(model, loss_func, dev_loader, optimizer=None):
    loss_val,corrects = 0.0,0.0
    for contexts, candidates, labels in dev_loader:
        contexts,labels = contexts.to(device),labels.to(device)
        preds = model(contexts, candidates.to(device))
        loss = loss_func(preds, labels)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds,labels = torch.argmax(preds, dim=1),torch.argmax(labels, dim=1)
        corrects += torch.sum(preds==labels).item()
        loss_val += loss.item() * contexts.size(0)

    dev_loss = loss_val / len(dev_loader.dataset)
    dev_acc = corrects / len(dev_loader.dataset)
    return dev_acc,dev_loss

def data(train_tensor, dev_tensor, test_tensor, bs):
    return (
        utils.data.DataLoader(train_tensor, batch_size=bs, shuffle=True),
        utils.data.DataLoader(dev_tensor, batch_size=bs),
        utils.data.DataLoader(test_tensor, batch_size=bs),
    )

if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config.data_path)

    # On récupère les données json retournant une liste de InputExample=texte
    train_examples = processor.get_train_examples(config.candidates_set_size)
    dev_examples = processor.get_dev_examples(config.candidates_set_size)
    test_examples = processor.get_test_examples(config.candidates_set_size)

    # Ensuite les données texte sont découpées en listes de jetons après le formatage=Tokenization
    train_dataset_tokens = processor.get_dataset_tokens(train_examples)
    dev_dataset_tokens = processor.get_dataset_tokens(dev_examples)
    test_dataset_tokens = processor.get_dataset_tokens(test_examples)
    
    if not os.path.exists(config.vocab_path) or config.update_vocab:
        processor.create_vocab(train_dataset_tokens, config.vocab_path)
    
    # On convertir un jeton en vecteur à l'aide du vocabulaire = embedding
    train_dataset_indices, vocab_size = processor.get_dataset_indices(train_dataset_tokens, config.vocab_path, config.vocab_size)
    dev_dataset_indices, _ = processor.get_dataset_indices(dev_dataset_tokens, config.vocab_path, config.vocab_size)
    test_dataset_indices, _ = processor.get_dataset_indices(test_dataset_tokens, config.vocab_path, config.vocab_size)
    config.vocab_size = vocab_size # Taille réelle de la liste de mots
    
    # Enfin les data loader sont chargés pour démarrer l'opération
    train_loader, dev_loader, test_loader = data(
        processor.create_tensor_dataset(train_dataset_indices, config.max_turn_num, config.max_seq_len),
        processor.create_tensor_dataset(dev_dataset_indices, config.max_turn_num, config.max_seq_len),
        processor.create_tensor_dataset(test_dataset_indices, config.max_turn_num, config.max_seq_len),
        config.batch_size)

    model = SMNModel(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()
    test_acc,test_loss = eval(model, nn.BCELoss(), test_loader)
    print(f"Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}")