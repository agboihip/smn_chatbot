# -*- coding: UTF-8 -*-​ 
import torch,copy,os
from smn_model import SMNModel
from data_processor import *

torch.manual_seed(512)

def train(model, train_loader, dev_loader, optimizer, loss_func, epochs, test_loader=None):
    best_val_acc,best_model_params = 0.0,copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        train_acc,train_loss = eval(model, loss_func, train_loader, optimizer)

        if epoch % 20 == 0:
            print(f"----------epoch/epochs: {epoch}/{epochs}----------")
            print(f"Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.2f}")

            model.eval()
            val_acc,val_loss = eval(model, loss_func, dev_loader)
            print(f"Dev Loss: {val_loss:.2f}, Dev Acc: {val_acc:.2f}")
            if val_acc > best_val_acc: best_val_acc,best_model_params = val_acc,copy.deepcopy(model.state_dict())
    if test_loader:
        model.eval()
        test_acc,test_loss = eval(model, loss_func, test_loader)
        print(f"Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}")
    model.load_state_dict(best_model_params)
    return model

if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config.data_path)

    # On récupère les données json retournant une liste de InputExample=texte
    train_examples = processor.get_train_examples(config.candidates_set_size)
    dev_examples = processor.get_dev_examples(config.candidates_set_size)

    # Ensuite les données texte sont découpées en listes de jetons après le formatage=Tokenization
    train_dataset_tokens = processor.get_dataset_tokens(train_examples)
    dev_dataset_tokens = processor.get_dataset_tokens(dev_examples)
    
    if not os.path.exists(config.vocab_path) or config.update_vocab:
        processor.create_vocab(train_dataset_tokens, config.vocab_path)
    
    # On convertir un jeton en vecteur à l'aide du vocabulaire = embedding
    train_dataset_indices, vs = processor.get_dataset_indices(train_dataset_tokens, config.vocab_path, config.vocab_size)
    dev_dataset_indices, _ = processor.get_dataset_indices(dev_dataset_tokens, config.vocab_path, config.vocab_size)
    
    # Enfin les data loader sont chargés pour démarrer l'opération
    train_loader = processor.create_tensor_dataset(train_dataset_indices, config.max_turn_num, config.max_seq_len,config.batch_size,True)
    dev_loader = processor.create_tensor_dataset(dev_dataset_indices, config.max_turn_num, config.max_seq_len,config.batch_size)

    config.vocab_size = vs # Taille réelle de la liste de mots
    model, loss_func = SMNModel(config).to(device), torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)

    model = train(model, train_loader, dev_loader, optimizer, loss_func, config.epochs)
    torch.save(model.state_dict(), config.model_save_path)
