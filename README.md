# TRD_FSL

This is the implementation of the paper [[Pre-trained Token-replaced Detection Model as  Few-shot Learner](https://arxiv.org/abs/2203.03235)]

## Requirements

Please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

## Datasets

The data file includes 16 processed datasets produced by the code in [LM-BFF](https://github.com/princeton-nlp/LM-BFF). Each dataset has 5 distinct training and development splits, randomly selected from the original training set using a fixed set of seeds. For additional information on the datasets, please refer to our [paper](https://arxiv.org/abs/2203.03235).

## Config

You can set parameters for each task in config.py. To illustrate some of the key parameters, we use SST-2 as an example.

```
 'SST-2': {
	'task_name': 'sst2', # task name
 	'template': {
   		'sentence': 'It was great terrible .',  # designed template and label description words
	        'class_index': {'0': 3, '1': 2,}, # real labels to indexs of label description words('0'->terrible, '1'->great)
	        'position': 2, # 0/1/2 means the prompt will be added before/between/after the sentence(s)
        	},
		    
	'data_format':{
		'suffix': 'tsv', # file type, tsv/csv
		'label_col': 1, # which column is the label
		'sentence_col': [0], # which columns are the input sentences
		'head': True, # does it have the head row
         	},
		       
         'metric': 'accuracy' # evaluation metric
         }
```

## Train & Evaluate

```
python train.py
```

The file results.txt will record the performance of each task.

### Citation

```
@inproceedings{li2022pre,
  title={Pre-trained Token-replaced Detection Model as Few-shot Learner},
  author={Li, Zicheng and Li, Shoushan and Zhou, Guodong},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={3274--3284},
  year={2022}
}
```
