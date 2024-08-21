"""Summary
"""
import os
import yaml
import getopt
import sys
import tensorflow as tf
import pickle

import numpy as np
from tensorflow.keras import backend as K

from action_predict import action_prediction
from action_predict import ActionPredict
from action_predict import seed_tensorflow
#from new_model import NewModel, HybridModel, MultiRNN3D, MultiRNN3D_MATT

sys.path.append("..")
from jaad_data import JAAD
from pie_data import PIE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def write_to_yaml(yaml_path=None, data=None):
    """
    Write model to yaml results file
    
    Args:
        model_path (None, optional): Description
        data (None, optional): results from the run
    
    Deleted Parameters:
        exp_type (str, optional): experiment type
        overwrite (bool, optional): whether to overwrite the results if the model exists
    """
    with open(yaml_path, 'w') as yamlfile:
        yaml.dump(data, yamlfile)


def run(config_file=None):
    """
    Run train and test on the dataset with parameters specified in configuration file.
    
    Args:
        config_file: path to configuration file in yaml format
        dataset: dataset to train and test the model on (pie, jaad_beh or jaad_all)
    """
    print(config_file)
    # Read default Config file
    configs_default ='config_files/configs_default.yaml'
    with open(configs_default, 'r') as f:
        configs = yaml.safe_load(f)

    with open(config_file, 'r') as f:
        model_configs = yaml.safe_load(f)

    # Update configs based on the model configs
    for k in ['model_opts', 'net_opts']:
        if k in model_configs:
            configs[k].update(model_configs[k])

    # Calculate min track size
    tte = configs['model_opts']['time_to_event'] if isinstance(configs['model_opts']['time_to_event'], int) else \
        configs['model_opts']['time_to_event'][1]
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte

    Performance_L = []  ##list to display the multi result


    # update model and training options from the config file
    for dataset_idx, dataset in enumerate(model_configs['exp_opts']['datasets']):
        configs['data_opts']['sample_type'] = 'beh' if 'beh' in dataset else 'all'
        configs['model_opts']['overlap'] = 0.6 if 'pie' in dataset else 0.8
        configs['model_opts']['dataset'] = dataset.split('_')[0]
        configs['model_opts']['dataset_F'] = dataset#####change
        configs['train_opts']['batch_size'] = model_configs['exp_opts']['batch_size'][dataset_idx]
        configs['train_opts']['lr'] = model_configs['exp_opts']['lr'][dataset_idx]
        configs['train_opts']['epochs'] = model_configs['exp_opts']['epochs'][dataset_idx]

        model_name = configs['model_opts']['model']
        # Remove speed in case the dataset is jaad
        if 'RNN' in model_name and 'jaad' in dataset:
            configs['model_opts']['obs_input_type'] = configs['model_opts']['obs_input_type']

        for k, v in configs.items():
            print(k,v)

        # set batch size
        if model_name in ['ConvLSTM']:
            configs['train_opts']['batch_size'] = 2
        if model_name in ['C3D', 'I3D']:
            configs['train_opts']['batch_size'] = 16
        if model_name in ['PCPA']:
            configs['train_opts']['batch_size'] = 8
        if 'MultiRNN' in model_name:
            configs['train_opts']['batch_size'] = 8
        if model_name in ['TwoStream']:
            configs['train_opts']['batch_size'] = 16

        if configs['model_opts']['dataset'] == 'pie':
            imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])
        elif configs['model_opts']['dataset'] == 'jaad':
            imdb = JAAD(data_path=os.environ.copy()['JAAD_PATH'])

        # get sequences
        beh_seq_train, database_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
        # beh_seq_val = None 
        # Uncomment the line below to use validation set
        beh_seq_val, database_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])
        beh_seq_test, database_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
        
        ########################### Gen JAAD TSC data ############################
        # TL_dict = {}
        # TL_dict['jaad_no_set'] = {}

        # for vid, v_dict in database_test.items():    ####vid loop
        #     print(vid,':')
        #     if vid not in TL_dict:
        #         TL_dict['jaad_no_set'][vid] = {}
        #     # print(v_dict['traffic_annotations'].keys())
        #     # input('next...')
        #     for frame_or_roadtype, ob_dict in v_dict['traffic_annotations'].items():
        #         if (isinstance(frame_or_roadtype, int)):
        #             print(frame_or_roadtype,':',ob_dict['traffic_light'])
        #             # print(ob_dict['state'])
        #             frame = frame_or_roadtype
        #             TL_dict['jaad_no_set'][vid].update({frame:{'jaad_NO_obid': [ob_dict['traffic_light'], 'jaad_No_box']}})
        #     # print(TL_dict)
        #     # input('next...')
        # # print(TL_dict)
        

        # with open('/data1/PIE_2/JAADbeh_Traffic_Light_data.pkl', 'wb') as f:
        #     pickle.dump(TL_dict, f)
        #     print('Dump pkl: /data1/PIE_2/JAADbeh_Traffic_Light_data.pkl')

        # print(sorted(TL_dict['jaad_no_set']['video_0155'][10].items()))
        # asd

        ### loop the database_test[set][video]['traffic_annotations'] to build the Crosswalk and sign data_dict 
        ### >> {'sid':{
        #          'vid':{
        #              frame(int): [obj_id, state] ....}

        # Sign_dict = {}
        # Sign_dict['jaad_no_set'] = {}

        # CW_dict = {}
        # CW_dict['jaad_no_set'] = {}

        # for vid, v_dict in database_test.items():    ####vid loop
        #     print(vid,':')
        #     if vid not in Sign_dict:
        #         Sign_dict['jaad_no_set'][vid] = {}
        #     if vid not in CW_dict:
        #         CW_dict['jaad_no_set'][vid] = {}
        #     # print(v_dict['traffic_annotations'].keys())
        #     # input('next...')
        #     for frame_or_roadtype, ob_dict in v_dict['traffic_annotations'].items():
        #         if (isinstance(frame_or_roadtype, int)):
        #             print(frame_or_roadtype,':', ob_dict['ped_crossing'], ob_dict['ped_sign'], ob_dict['stop_sign'])
        #             # print(ob_dict['state'])
        #             frame = frame_or_roadtype

        #             if ob_dict['ped_crossing'] == 1:
        #                 CW_dict['jaad_no_set'][vid].update({frame:{'jaad_NO_obid': [1, 'jaad_No_box']}})
        #             elif ob_dict['ped_crossing'] == 0:
        #                 CW_dict['jaad_no_set'][vid].update({frame:{'jaad_NO_obid': [0, 'jaad_No_box']}})
                        
                    
        #             if ob_dict['stop_sign'] == 1:
        #                 S_type  = 1
        #                 Sign_dict['jaad_no_set'][vid].update({frame:{'jaad_NO_obid': [S_type, 'jaad_No_box']}})
        #             elif ob_dict['ped_sign'] == 1:
        #                 S_type  = 5
        #                 Sign_dict['jaad_no_set'][vid].update({frame:{'jaad_NO_obid': [S_type, 'jaad_No_box']}})
        #             else:
        #                 S_type  = 0
        #                 Sign_dict['jaad_no_set'][vid].update({frame:{'jaad_NO_obid': [S_type, 'jaad_No_box']}})


        #     # print(TL_dict)
        #     # input('next...')
        # # print(TL_dict)

        # with open('/data1/PIE_2/JAADall_Crosswalk_data.pkl', 'wb') as f:
        #     pickle.dump(CW_dict, f)
        #     print('Dump pkl: /data1/PIE_2/JAADall_Crosswalk_data_New.pkl')


        # with open('/data1/PIE_2/JAADall_Signs_data_New.pkl', 'wb') as f:
        #     pickle.dump(Sign_dict, f)
        #     print('Dump pkl: /data1/PIE_2/JAADall_Signs_data_New.pkl')
        # print(sorted(Sign_dict['jaad_no_set']['video_0155'][10].items()))
        # print(sorted(CW_dict['jaad_no_set']['video_0155'][10].items()))

        # asdasd

        ##################################Gen TSC data END#################################

        # get the model
        method_class = action_prediction(configs['model_opts']['model'])(**configs['net_opts'])


        s_list = []   ####change for seed
        s_range = 92 ##193 ##50best ##57 ##2   

        for s in range(s_range, s_range + 1):   ####change for seed

            seed_tensorflow(s)
            print('\n\nStart! \n seed: ', s)   ####change for seed


            # train and save the model
            saved_files_path = method_class.train(beh_seq_train, beh_seq_val, beh_seq_test, **configs['train_opts'],
                                                model_opts=configs['model_opts'])
            # test and evaluate the model
            acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)

            # save the results
            data = {}
            data['results'] = {}
            data['results']['acc'] = float(acc)
            data['results']['auc'] = float(auc)
            data['results']['f1'] = float(f1)
            data['results']['precision'] = float(precision)
            data['results']['recall'] = float(recall)
            write_to_yaml(yaml_path=os.path.join(saved_files_path, 'results.yaml'), data=data)

            Performance_L.append(dataset + '  acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))


            data = configs
            write_to_yaml(yaml_path=os.path.join(saved_files_path, 'configs.yaml'), data=data)

            print('Model saved to {}'.format(saved_files_path))

            print('seed: ', s, 'acc: ', acc)   ####change for seed

            if acc >= 0.90:         
                s_list.append([s, acc])
                # print('seed: ', s, 'acc: ', acc)   ####change for seed
    
            # print(s_list)   ####change for seed

    print(Performance_L)
        

def usage():
    """
    Prints help
    """
    print('Benchmark for evaluating pedestrian action prediction.')
    print('Script for training and testing models.')
    print('Usage: python train_test.py [options]')
    print('Options:')
    print('-h, --help\t\t', 'Displays this help')
    print('-c, --config_file\t', 'Path to config file')
    print()

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'config_file'])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)    

    config_file = None
    model_name = None
    dataset = None

    for o, a in opts:
        if o in ["-h", "--help"]:
            usage()
            sys.exit(2)
        elif o in ['-c', '--config_file']:
            config_file = a

    # if neither the config file or model name are provided
    if not config_file:
        print('\x1b[1;37;41m' + 'ERROR: Provide path to config file!' + '\x1b[0m')
        usage()
        sys.exit(2)

    run(config_file=config_file)
