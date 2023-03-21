
# define datasets dir and the splits of every dataset
DATASETS_PATH = 'data/k-shot'
DATA_SEED = ['16-13', '16-21', '16-42', '16-87', '16-100']

# define the template, label description words and other parameters for every dataset
TASKS = {
        # SINGLE SENTENCE TASK
        'SST-2': { # dataset file name
            'task_name': 'sst2', # task name
            'template': {'sentence': 'It was great terrible .',  # designed template
                        'class_index': {'0': 3, '1': 2,}, # '0' and '1' are the real labels, 3 and 2 are corresponding indexs of label description words
                        'position': 2, # 0/1/2 mean the prompt will be added before/between/after the sentence(s)
                        },
            'data_format':{'suffix': 'tsv', # file type, tsv/csv
                            'label_col': 1, # which column is the label
                            'sentence_col': [0], # which columns are the input sentences
                            'head': True, # does it have the head row
                        },
            'metric': 'accuracy' # for evaluate
            },

        'sst-5':{
            'task_name': 'sst5',
            'template': {'sentence': 'It was terrible bad okay good great .', 
                        'class_index': {'0':2, '1':3, '2':4, '3':5, '4':6},
                        'position': 2,
                        },
            'data_format':{'suffix': 'csv',
                            'label_col': 0,
                            'sentence_col': [1],
                            'head': False,
                        },
            'metric': 'accuracy'
            },

        'mr':{
            'task_name': 'mr',
            'template': {'sentence': 'It was great terrible .', 
                        'class_index': {'0': 3, '1': 2,},
                        'position': 2,
                        }, 
            'data_format':{'suffix': 'csv',
                            'label_col': 0,
                            'sentence_col': [1],
                            'head': False,
                        },
            'metric': 'accuracy'
            },

        'cr':{
            'task_name': 'cr',
            'template': {'sentence': 'It was great terrible .', 
                        'class_index': {'0': 3, '1': 2,},
                        'position': 2,
                        }, 
            'data_format':{'suffix': 'csv',
                            'label_col': 0,
                            'sentence_col': [1],
                            'head': False,
                        },
            'metric': 'accuracy'
            },

        'mpqa':{
            'task_name': 'mpqa',
            'template': {'sentence': 'It was great terrible .', 
                        'class_index': {'0': 3, '1': 2,}, # 0是消极
                        'position': 2,
                        }, 
            'data_format':{'suffix': 'csv',
                            'label_col': 0,
                            'sentence_col': [1],
                            'head': False,
                        },
            'metric': 'accuracy'
            },

        'subj':{
            'task_name': 'subj',
            'template': {'sentence': 'This is subjective objective .', 
                        'class_index': {'0': 2, '1': 3,},
                        'position': 2,
                        }, 
            'data_format':{'suffix': 'csv',
                            'label_col': 0,
                            'sentence_col': [1],
                            'head': False,
                        },
            'metric': 'accuracy'
            },

        'trec':{
            'task_name': 'trec',
            'template': {'sentence': 'Description Entity Expression Human Location Number :', 
                        'class_index': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
                        'position': 0,
                        }, 
            'data_format':{'suffix': 'csv',
                            'label_col': 0,
                            'sentence_col': [1],
                            'head': False,
                        },
            'metric': 'accuracy'
            },

        'CoLA':{
            'task_name': 'cola',
            'template': {'sentence': 'This is correct incorrect .', 
                        'class_index': {'0': 3, '1': 2,},
                        'position': 2,
                        }, 
            'data_format':{'suffix': 'tsv',
                            'label_col': 1,
                            'sentence_col': [3],
                            'head': False,
                        },
            'metric': 'matthews_correlation'
            },


        # DOUBLE SENTENCE TASKS
        'MNLI':{
            'task_name': 'mnli',
            'template': {'sentence': '? Yes Maybe No ,', 
                        'class_index': {'entailment': 1, 'neutral': 2, 'contradiction': 3},
                        'position': 1,
                        }, 
            'data_format':{'suffix': 'tsv',
                            'label_col': -1, 
                            'sentence_col': [8, 9],
                            'head': True,
                            'splits':{
                                'train': 'train',
                                'dev': 'dev_matched',
                                # 'test': 'test_mismatched', # for MNLI-MM
                                'test': 'test_matched'
                            }
                        },
            'metric': 'accuracy'
            }, 

        'SNLI':{
            'task_name': 'snli',
            'template': {'sentence': '? Yes Maybe No ,', 
                        'class_index': {'entailment': 1, 'neutral': 2, 'contradiction': 3},
                        'position': 1,
                        }, 
            'data_format':{'suffix': 'tsv',
                            'label_col': -1, 
                            'sentence_col': [7, 8],
                            'head': True,
                        },
            'metric': 'accuracy'
            }, 
        
        'QNLI':{
            'task_name': 'qnli',
            'template': {'sentence': '? Yes No ,', 
                        'class_index': {'entailment': 1, 'not_entailment': 2},
                        'position': 1,
                        }, 
            'data_format':{'suffix': 'tsv',
                            'label_col': -1, 
                            'sentence_col': [1, 2],
                            'head': True,
                        },
            'metric': 'accuracy'
            }, 

        'RTE':{
            'task_name': 'rte',
            'template': {'sentence': '? Yes No ,', 
                        'class_index': {'entailment': 1, 'not_entailment': 2},
                        'position': 1,
                        }, 
            'data_format':{'suffix': 'tsv',
                            'label_col': -1, 
                            'sentence_col': [1, 2],
                            'head': True,
                        },
            'metric': 'accuracy'
            }, 

        'MRPC':{
            'task_name': 'mrpc',
            'template': {'sentence': 'No Yes ,', 
                        'class_index': {'0': 0, '1': 1, }, 
                        'position': 1,
                        }, 
            'data_format':{'suffix': 'tsv',
                            'label_col': 0, 
                            'sentence_col': [3, 4],
                            'head': True,
                        },
            'metric': 'f1'
            }, 

        'QQP':{
            'task_name': 'qqp',
            'template': {'sentence': 'No Yes ,', 
                        'class_index': {'0': 0, '1': 1},
                        'position': 1,
                        }, 
            'data_format':{'suffix': 'tsv',
                            'label_col': -1, 
                            'sentence_col': [3, 4],
                            'head': True,
                        },
            'metric': 'f1'
            }, 

        'STS-B':{
                'task_name': 'stsb',
                'task_type': 'regression',
                'template': {'sentence': 'Yes No ,', 
                            'class_index': {'Yl': 1, 'Yu': 0},
                            'position': 1,
                            }, 
                'data_format':{'suffix': 'tsv',
                                'label_col': -1, 
                                'sentence_col': [7, 8],
                                'head': True,
                                'Vl_Vu': (0, 5)
                            },
                'metric': 'pearsonr'
                }, 
        }

for t in TASKS:
    TASKS[t]['seed'] = 1234
    TASKS[t]['model_name'] = 'google/electra-base-discriminator'
    # TASKS[t]['model_name'] = 'google/electra-large-discriminator'
    TASKS[t]['batch_size'] = [4, 8]
    TASKS[t]['learning_rate'] = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    TASKS[t]['weight_decay'] = 2e-3
    TASKS[t]['epsilon'] = 1e-8
    if t in ['cr', 'mpqa', 'RTE', 'MRPC']:
        TASKS[t]['num_epoch'] = 15
    else:
        TASKS[t]['num_epoch'] = 30

    if t in ['MNLI']:
        TASKS[t]['max_length'] = 512
    else:
        TASKS[t]['max_length'] = 256

