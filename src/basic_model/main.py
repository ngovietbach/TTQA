from src.basic_model.model_architecture.model_one_sequence_learning import LSTMCNNWord
from src.basic_model.model_architecture.model_pair_infersent import LSTMCNNWordInferSent
from src.basic_model.model_architecture.model_pair_bdaf import BiDAF

from src.basic_model.train_basic import Trainer
from module_dataset.preprocess_dataset.handle_dataloader_basic import *

from transformers import BertTokenizer
from transformers import BertConfig



def train_model_basic(cf_common, cf_model):
    path_data = cf_common['path_data']
    path_data_train = cf_common['path_data_train']
    path_data_test = cf_common['path_data_test']

    type_model = cf_common['type_model']
    model = None
    data_train_iter = None
    data_test_iter = None
    data = None

    # first load data
    if "one_sequence" in type_model:
        data = load_data_word_lstm_char(path_data,
                                        path_data_train,
                                        path_data_test,
                                        device_set=cf_common['device_set'],
                                        min_freq_word=cf_common['min_freq_word'],
                                        min_freq_char=cf_common['min_freq_char'],
                                        batch_size=cf_common['batch_size'],
                                        cache_folder=cf_common['cache_folder'],
                                        name_vocab=cf_common['name_vocab'],
                                        path_vocab_pre_built=cf_common['path_vocab_pre_built']
                                        )

        data_train_iter = data['iters'][0]

        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

    if "pair_sequence" in type_model:
        data = load_data_pair_task(path_data,
                                   path_data_train,
                                   path_data_test,
                                   device_set=cf_common['device_set'],
                                   min_freq_word=cf_common['min_freq_word'],
                                   min_freq_char=cf_common['min_freq_char'],
                                   batch_size=cf_common['batch_size'],
                                   cache_folder=cf_common['cache_folder'],
                                   name_vocab=cf_common['name_vocab'],
                                   path_vocab_pre_built=cf_common['path_vocab_pre_built']
                                   )

        data_train_iter = data['iters'][0]

        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

    print("!!Load dataset done !!\n")

    if type_model == "one_sequence_lstm_cnn":
        model = LSTMCNNWord.create(cf_common['path_save_model'] + cf_common['folder_model'],
                                   cf_model,
                                   data['vocabs'],
                                   device_set=cf_common['device_set'])

    elif type_model == "pair_sequence_infer_sent":
        model = LSTMCNNWordInferSent.create(cf_common['path_save_model'] + cf_common['folder_model'],
                                            cf_model,
                                            data['vocabs'],
                                            device_set=cf_common['device_set'])

    elif type_model == "pair_sequence_bidaf":
        model = BiDAF.create(cf_common['path_save_model'] + cf_common['folder_model'],
                                            cf_model,
                                            data['vocabs'],
                                            device_set=cf_common['device_set'])

    trainer = Trainer(cf_common['path_save_model'] + cf_common['folder_model'],
                      model,
                      cf_model,
                      cf_common['prefix_model'],
                      cf_common['log_file'],
                      len(data['vocabs'][2]),
                      data_train_iter,
                      data_test_iter)

    trainer.train(cf_common['num_epochs'])




if __name__ == '__main__':
    cf_common = {
        "path_save_model": "save_model/",
        "path_data": "../module_dataset/dataset/dataset_split_with_preprocess/pair_sequence",
        "path_data_train": "train_origin_has_segment.csv",
        "path_data_test": "test_pair_sequence_has_segment.csv",
        "prefix_model": "pair_sequence_bidaf_has_sg_",
        "log_file": "log_pair_sequence_bidaf_has_sg_hsw_64_char_emb_64_drop_03.txt",
        "type_model": "pair_sequence_bidaf",
        "folder_model": "model_1",
        'path_checkpoint': "",
        "device_set": "cuda:0",
        "num_epochs": 30,
        "min_freq_word": 1,
        "min_freq_char": 5,
        "path_vocab_pre_built": "../module_dataset/dataset/vocab_build_all/vocab_baomoi_400_augment_has_sg_min_freq_2.pt",
        "cache_folder": None,
        "name_vocab": None,
        "sort_key": True,
        "batch_size": 32
    }

    cf_model_lstm_cnn_word = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 400,
        'char_embedding_dim': 32,
        'hidden_size_word': 32,
        'hidden_size_char_lstm': 16,
        'use_highway_char': False,
        'use_char_cnn': False,
        'dropout_cnn_char': 0.3,
        'D_cnn': '1_D',
        'char_cnn_filter_num': 5,
        'char_window_size': [2, 3],
        'use_last_as_ft': False,
        "option_last_layer": "max_pooling",
        "cnn_filter_num": 16,
        "window_size": [1],
        'dropout_cnn_word': 0.55,
        'dropout_rate': 0.35,
        'learning_rate': 0.0001,
        'weight_decay': 0
    }

    cf_model_infer_sent = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 400,
        'char_embedding_dim': 32,
        'hidden_size_word': 32,
        'hidden_size_char_lstm': 16,

        'use_highway': False,
        'use_char_cnn': False,
        'dropout_cnn_char': 0.3,
        'D_cnn': '1_D',
        'char_cnn_filter_num': 5,
        'char_window_size': [2, 3],

        "option_last_layer": "max_pooling",
        "cnn_filter_num": 16,
        "window_size": [1],
        'dropout_cnn_word': 0.55,

        'hidden_layer_1_dim': 1600,
        'hidden_layer_2_dim': 512,

        'weight_class': [1, 1],
        'dropout_rate': 0.55,
        'learning_rate': 0.0001,
        'weight_decay': 0
    }

    cf_model_bidaf = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 400,
        'char_embedding_dim': 64,
        'hidden_size_word': 64,
        'hidden_size_char_lstm': 8,

        'use_highway': True,
        'use_char_cnn': True,
        'dropout_cnn_char': 0.3,
        'char_cnn_filter_num': 1,
        'char_window_size': [3],

        "use_modeling_in_last": False,
        'num_layer_modeling_lstm': 2,
        "option_last_layer": "max_pooling",
        "cnn_filter_num": 2,
        "window_size": [3],
        'dropout_cnn': 0.55,

        'weight_class': [0.6, 1],
        'dropout_rate': 0.3,
        'learning_rate': 0.0001,
        'weight_decay': 0
    }

    train_model_basic(cf_common, cf_model_bidaf)

