from torch import cuda,load,nn
from smn_model import SMNModel
from data_processor import *

cuda.manual_seed(512)
if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config.data_path)

    test_examples = processor.get_test_examples(config.candidates_set_size)
    test_dataset_tokens = processor.get_dataset_tokens(test_examples)
    test_dataset_indices, vocab_size = processor.get_dataset_indices(test_dataset_tokens, config.vocab_path, config.vocab_size)
    config.vocab_size = vocab_size # Taille réelle de la liste de mots
    
    test_loader = processor.create_tensor_dataset(test_dataset_indices, config.max_turn_num, config.max_seq_len,config.batch_size)
    model = SMNModel(config).to(device)
    model.load_state_dict(load(config.model_save_path))
    model.eval()
    test_acc,test_loss = eval(model, nn.BCELoss(), test_loader)
    print(f"Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}")