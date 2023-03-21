import csv
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import ElectraTokenizer
import os

REPLACED_ID = 1
ORIGINAL_ID = 0

class FewshotDataset(Dataset):
    def __init__(self, data_path, split, template, model_name, max_seq_length, data_format, regression=False): 
        super(FewshotDataset, self).__init__()

        self.regression = regression
        self.template_sentence = template['sentence'].split()
        self.position = template['position']
        self.class_index = template['class_index']
        self.class_index_lst = sorted(list(self.class_index.values())) # 哪些索引是special token

        self.max_seq_length = max_seq_length
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)

        # read all samples, format: [[(tokens),(label)], []]
        self.examples = self._get_labeled_sentences(data_path, split, data_format)
    
    # read all samples
    def _get_labeled_sentences(self, data_path, split, data_format):
        if 'splits' in data_format:
            split = data_format['splits'][split]
        # read file
        file_suffix = data_format['suffix']
        if file_suffix == 'tsv':
            f = open(os.path.join(data_path, split+'.tsv'))
            reader = csv.reader(f, delimiter='\t', quoting=3) 
        elif file_suffix == 'csv':
            f = open(os.path.join(data_path, split+'.csv'))
            reader = csv.reader(f) 
        else:
            raise f'{file_suffix} data format not support'


        # collect sentences, format：[[tokens, label, special_idx, sentence_label], []]
        # special_idx record the label description words' index
        sentences = [] 
        label_col = data_format['label_col']
        label_col = label_col[split] if isinstance(label_col, Dict) else label_col
        sentence_col = data_format['sentence_col']
        sentence_num = len(sentence_col)
        
        for idx, line in enumerate(reader):
            if data_format['head'] and idx==0: continue # skip head
            
            # construct train tokens
            if sentence_num == 1: # one sentence input
                # read sentence
                original_sentence = line[sentence_col[0]].split()
                # add prompt to the input
                if self.position == 0:
                    input_sentence = self.template_sentence + original_sentence
                    template_idx =  0
                elif self.position == 2:
                    input_sentence = original_sentence + self.template_sentence
                    template_idx =  len(original_sentence)
                else:
                    raise 'position not support for single sentence input'
            elif sentence_num == 2: # two sentence input
                # read sentence
                original_sentence_first = line[sentence_col[0]].split()
                original_sentence_second = line[sentence_col[1]].split()

                 # add prompt
                if self.position == 1:
                    # if ? . ,， remove the last character of the first sentence
                    if self.template_sentence[0] in ['?', '.', ',']:
                        original_sentence_first[-1] = original_sentence_first[-1][:-1]

                    # lower the first word
                    original_sentence_second[0] = original_sentence_second[0].lower()

                    input_sentence = original_sentence_first\
                          + self.template_sentence + original_sentence_second
                    template_idx =  len(original_sentence_first)
                else:
                    raise 'position not support for double sentence input'
            else:
                raise 'not support for multi sentence input'

            # construct train label
            special_token_idx = [i+template_idx for i in self.class_index_lst]
            sentence_label = line[label_col]
            if self.regression: # regression task
                Vl, Vu = data_format['Vl_Vu']
                Vu_sub_Vl = Vu - Vl

                # compute the train label for yu and yl
                label = [ORIGINAL_ID for i in range(len(input_sentence))]
                label_yu = (float(sentence_label) - Vl) / Vu_sub_Vl
                label_yl =  (Vu - float(sentence_label)) / Vu_sub_Vl
                
                # set label of the two label description words
                yu_id = self.class_index['Yu'] + template_idx 
                yl_id = self.class_index['Yl'] + template_idx 
                label[yu_id] = 1 - label_yu
                label[yl_id] = 1 - label_yl

            else: # classification task
                label = [REPLACED_ID if i in special_token_idx else ORIGINAL_ID
                            for i in range(len(input_sentence))]
                
                keep_id = self.class_index[sentence_label] + template_idx # the index need to set original
                label[keep_id] = ORIGINAL_ID

            sentences.append((input_sentence, label, special_token_idx, sentence_label))

        f.close()
        return sentences

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):   
        [sentence, label, special_token_idx, sentence_label] = self.examples[index]
        
        # get input_ids, full_labels, full_labels_mask_local
        input_ids = [self.tokenizer.cls_token_id]
        full_labels = [ORIGINAL_ID]
        full_labels_mask_local = [0.0]
        full_special_token_idx = []
        for i, (word, l) in enumerate(zip(sentence, label)):
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            input_ids += word_ids

            word_ids_length = len(word_ids)
            full_labels += [l] * word_ids_length

            if i in special_token_idx:# only kepp the first sub_word
                full_special_token_idx.append(len(full_labels_mask_local))
                full_labels_mask_local += [1.0 if i==0 else 0.0 for i in range(word_ids_length)]
            else:
                full_labels_mask_local += [0.0] * word_ids_length

        input_ids += [self.tokenizer.sep_token_id]
        full_labels += [ORIGINAL_ID]
        full_labels_mask_local += [0.0]

        # If it is too long (exceed max_length)
        if len(input_ids) > self.max_seq_length:
            max_idx = max(full_special_token_idx)
            if max_idx < (self.max_seq_length - 1): # max_seq_length - 2
                print('remain preceding max_seq_length characters')
                # clip
                input_ids = input_ids[: self.max_seq_length]
                full_labels = full_labels[: self.max_seq_length]
                full_labels_mask_local = full_labels_mask_local[: self.max_seq_length]

                input_ids[-1] = self.tokenizer.sep_token_id
                full_labels[-1] = ORIGINAL_ID
                full_labels_mask_local[-1] = 0.0
            else:# keep the special tokens
                print('remain middle max_seq_length characters')
                half_length = self.max_seq_length // 2
                s = max_idx - half_length
                e = max_idx + half_length

                input_ids = input_ids[s: e]
                full_labels = full_labels[s: e]
                full_labels_mask_local = full_labels_mask_local[s: e]

                input_ids[0] = self.tokenizer.cls_token_id
                full_labels[0] = ORIGINAL_ID
                full_labels_mask_local[0] = 0.0
                input_ids[-1] = self.tokenizer.sep_token_id
                full_labels[-1] = ORIGINAL_ID
                full_labels_mask_local[-1] = 0.0


        # pad 0 to max_seq_length
        pad = lambda x: x + [0] * (self.max_seq_length - len(x))
        
        full_labels = pad(full_labels)
        full_labels_mask_local = pad(full_labels_mask_local)

        segment_ids = pad([0] * len(input_ids)) # sentence A set 0
        input_mask = pad([1] * len(input_ids))
        input_ids = pad(input_ids)
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        assert len(full_labels) == self.max_seq_length
        assert len(full_labels_mask_local) == self.max_seq_length

   
        return {
            "sentence": ' '.join(sentence),
            "sentence_label": sentence_label,
            
            "input_ids": torch.tensor(input_ids), # input
            "input_mask": torch.tensor(input_mask), # input mask
            "segment_ids": torch.tensor(segment_ids), # segment_id

            "full_labels":torch.tensor(full_labels),
            "full_labels_mask_local":torch.tensor(full_labels_mask_local),
        }


if __name__ == '__main__':
    template = ' it is great terrible .'
    cls_index = { # 每个类别，对应转换哪个索引保持为original
        '0': -2, # 负面情感
        '1': -3  # 正面
    }
    train_dataset = FewshotDataset(data_path='yelp', 
                                    split='train',
                                    template=template,
                                    class_index=cls_index,
                                    model_name='google/electra-small-discriminator',
                                    max_seq_length=40)
    dev_loader = DataLoader(train_dataset, batch_size=1)
    for batch in dev_loader:
        for k, v in batch.items():
           print(k)
           print(v)
    
        print()
    exit(0)

