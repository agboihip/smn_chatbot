# -*- coding: UTF-8 -*-​ 
import collections,unicodedata
import json,torch,numpy as np

torch.manual_seed(1024)
torch.cuda.manual_seed(1024)


class InputExample:
    def __init__(self, guid, context, candidate, label):
        self.guid = guid
        self.context = context
        self.candidate = candidate
        self.label = label

class DataProcessor:
    def __init__(self, data_path):
        self.datasets = dict()
        for data_type in data_path.keys():
            with open(data_path[data_type], "r", encoding="utf-8") as f:
                data = json.load(f)
                self.datasets[data_type] = data

    def _convert_to_unicode(self, text):
        """将输入转为unicode编码"""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def _is_whitespace(self, char):
        """判断字符是否为空格"""
        # 将'\t'、'\n'、'\r'当作空格，用于将文本切分为token
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        
        return cat == "Zs" # Separator, Space

    def _is_control(self, char):
        """判断字符是否为控制字符"""
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        return cat.startswith("C") # Control

    def _is_punctuation(self, char):
        """判断字符是否为标点符号"""
        cp = ord(char)
        # 不在unicode中的字符，如： "^", "$"和"`" 用SSCII判断
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        return cat.startswith("P") # Punctuation
    
    def _clean_text(self, text):
        """删除无效字符, 将\t\r\n等字符用空格替代"""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _whitespace_tokenize(self, text):
        """用空格将文本进行切分"""
        text = text.strip()
        if not text: return []
        tokens = text.split()
        return tokens
    
    def _strip_accents(self, token):
        """去除token中的重音"""
        output,token = [],unicodedata.normalize("NFD", token)
        
        for char in token:
            cat = unicodedata.category(char)
            if cat == "Mn": continue
            output.append(char)
        return "".join(output)

    def _split_on_punc(self, token):
        """对token根据标点符号进行再切分，并将标点符号作为一个单独的token"""
        output,chars = [],list(token)
        i,start_new_word = 0,True
        
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word: output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _create_examples(self, datas, data_type, n):
        """Envelopper le json d'entrée dans une structure de données input_example"""
        examples = []
        for data in datas:
            candidates,contexts = [],[text["utterance"] for text in data["messages-so-far"]]
            label,label_idx = [], None
            if "options-for-correct-answers" in data:
                gt = data["options-for-correct-answers"][0]["candidate-id"]
                for candidate_idx, candidate in enumerate(data["options-for-next"]):
                    if candidate["candidate-id"] == gt:
                        label.append(1)
                        label_idx = candidate_idx
                        candidates.append(candidate["utterance"])
                        break
                if label_idx is None: raise ValueError(f"Data: {data['example-id']} has no label")

            # Exemples négatifs choisis au hasard
            neg_indices = list(range(len(data["options-for-next"]))) #[idx for idx in range(label_idx)] + [idx for idx in range(label_idx+1, len(data["options-for-next"]))] 
            if label_idx: neg_indices.remove(label_idx) # Sauter la catégorie positive
            np.random.shuffle(neg_indices) 
            neg_indices = neg_indices[:n-1]
            for candidate_idx in neg_indices:
                candidates.append(data["options-for-next"][candidate_idx]["utterance"])
                label.append(0)

            # Melanger candidates, label à la fois
            candidates_label = list(zip(candidates, label))
            np.random.shuffle(candidates_label)
            candidates, label = zip(*candidates_label)

            guid = "%s-%s" % (data_type, data["example-id"])
            examples.append(InputExample(guid, contexts, list(candidates), list(label)))
        return examples

    def get_train_examples(self, n):
        return self._create_examples(self.datasets["train"], "train", n)
    
    def get_dev_examples(self, n):
        return self._create_examples(self.datasets["dev"], "dev", n)

    def get_test_examples(self, n):
        return self._create_examples(self.datasets["test"], "test", n)
    
    def get_dataset_tokens(self, examples):
        """Découpage du texte en listes de jetons"""
        datasets = []
        for example in examples:
            contexts,contexts_tokens = example.context,[]
            candidates,candidates_tokens = example.candidate,[]

            
            for context in contexts:
                context = self._convert_to_unicode(context)
                context = self._clean_text(context)
                tokens = self._whitespace_tokenize(context)
                post_tokens = []
                for token in tokens:
                    token = self._strip_accents(token.lower())
                    post_tokens.extend(self._split_on_punc(token))
                contexts_tokens.append(post_tokens)
                
            
            for candidate in candidates:
                candidate = self._convert_to_unicode(candidate)
                candidate = self._clean_text(candidate)
                tokens = self._whitespace_tokenize(candidate)
                post_tokens = []
                for token in tokens:
                    token = self._strip_accents(token.lower())
                    post_tokens.extend(self._split_on_punc(token))
                candidates_tokens.append(post_tokens)

            datasets.append(
                InputExample(guid=example.guid, context=contexts_tokens, candidate=candidates_tokens, label=example.label)
            )
        return datasets

    def create_vocab(self, datasets, vocab_path):
        """Création de listes de mots avec les tokens de l'ensemble de formation"""
        count_dict = dict()
        for dataset in datasets:
            contexts,candidates = dataset.context,dataset.candidate

            for context in contexts:
                for token in context:
                    if token in count_dict:
                        count_dict[token] += 1
                    count_dict[token] = 0

            for candidate in candidates:
                for token in candidate: count_dict[token] = int(token in count_dict)
        token_count_sorted = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
        
        with open(vocab_path, "w") as f:
            f.write("<pad>\n") # Remplissage des phrases
            f.write("<unk>\n") # Mots qui n'apparaissent pas dans la liste des mots représentatifs
            for item in token_count_sorted: f.write(item[0] + "\n")
    
    def _load_vocab(self, vocab_path, vocab_size):
        token2index = dict()
        with open(vocab_path, "r", encoding="utf-8") as f:
            idx = 0
            for token in f.readlines():
                token = self._convert_to_unicode(token)
                token = token.strip()
                token2index[token] = idx
                idx += 1
                if idx > vocab_size: break
        return token2index, min(idx, vocab_size)
    
    def get_dataset_indices(self, datasets, vocab_path, vocab_size):
        """Convertir un jeton en index en utilisant un dictionnaire vocab"""
        vocab, vocab_size = self._load_vocab(vocab_path, vocab_size)
        dataset_indices = []
        for dataset in datasets:
            contexts,candidates = dataset.context,dataset.candidate   
            context_indices ,candidate_indices = [],[]

            for context in contexts:
                indices = [vocab[token] if token in vocab else vocab["<unk>"] for token in context] #for token in context: if token in vocab: indices.append(vocab[token]) else: indices.append(vocab["<unk>"])
                context_indices.append(indices)
            
            for candidate in candidates:
                indices = [vocab[token] if token in vocab else vocab["<unk>"] for token in candidate] #for token in context: if token in vocab: indices.append(vocab[token]) else: indices.append(vocab["<unk>"])
                candidate_indices.append(indices)

            dataset_indices.append(
                InputExample(guid=dataset.guid, context=context_indices, candidate=candidate_indices, label=dataset.label)
            )
        return dataset_indices, vocab_size
        
    def create_tensor_dataset(self, datas, max_turn_num, max_seq_len):
        """Création d'un ensemble de données"""
        all_contexts,all_candidates = [],[]
        all_labels = []
        for data in datas:
            contexts,new_contexts = data.context,[]
            candidates,new_candidates = data.candidate,[]

            if len(contexts) > max_turn_num: contexts = contexts[-max_turn_num:]
            # Les tours de dialogue insuffisants sont complétés par tous les zéros <pad>.
            contexts += [[0] * max_seq_len] * (max_turn_num - len(contexts))
            for context in contexts:
                if len(context) > max_seq_len: context = context[-max_seq_len:]
                context += [0] * (max_seq_len - len(context)) #0<pad> remplissage à la fin du texte court
                new_contexts.append(context)
            
            
            for candidate in candidates:
                if len(candidate) > max_seq_len: candidate = candidate[-max_seq_len:]
                candidate += [0] * (max_seq_len - len(candidate))
                new_candidates.append(candidate)
            
            all_contexts.append(new_contexts)
            all_candidates.append(new_candidates)
            all_labels.append(data.label)

        all_labels = torch.FloatTensor(all_labels)
        all_contexts,all_candidates = torch.LongTensor(all_contexts),torch.LongTensor(all_candidates)

        return torch.utils.data.TensorDataset(all_contexts, all_candidates, all_labels)