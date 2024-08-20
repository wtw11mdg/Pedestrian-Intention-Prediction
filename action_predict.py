
import time
import yaml
import wget
import cv2
from utils import *
from base_models import AlexNet, C3DNet, convert_to_fcn
from base_models import I3DNet
from tensorflow.keras.layers import Input, Concatenate, Dense, Conv1D
from tensorflow.keras.layers import GRU, LSTM, GRUCell
from tensorflow.keras.layers import Dropout, LSTMCell, RNN
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Average, Add
from tensorflow.keras.layers import ConvLSTM2D, Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, dot, concatenate, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import argparse
import sys
sys.path.append("..")
from transformer_TF2.model import *
from focal_loss import BinaryFocalLoss

from whenet import WHENet
from yolo_v3.yolo_postprocess import YOLO
from utils_Head import draw_axis
from CBAM_attention3D import channel_attention
from vit_keras_local.vit_keras_local import utils

from skimage import io
from face_detection import RetinaFace
from sixdrepnet import SixDRepNet

def seed_tensorflow(seed = 19):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    tf.random.set_seed(seed)  # tf cpu fix seed
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first
    
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# seed_tensorflow(193)

# TODO: Make all global class parameters to minimum , e.g. no model generation
class ActionPredict(object):
    """
        A base interface class for creating prediction models
    """

    def __init__(self,
                 global_pooling='avg',
                 regularizer_val=0.0001,
                 backbone='vgg16',
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Pooling method for generating convolutional features
            regularizer_val: Regularization value for training
            backbone: Backbone for generating convolutional features
        """
        # Network parameters
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)
        self._global_pooling = global_pooling
        self._backbone = backbone
        self._generator = None # use data generator for train/test 

    # Processing images anf generate features
    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids,
                                     feature_type, ###change
                                     dataset_F,   ###change
                                     save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     process=True,
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            process: Whether process the raw images using a neural network
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, ".format(data_type, crop_type, crop_mode,
                                       save_path))
        preprocess_dict = {'vgg16': vgg16.preprocess_input, 'resnet50': resnet50.preprocess_input}
        backbone_dict = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50}
        
        preprocess_input = preprocess_dict.get(self._backbone, None)
        if process:
            assert (self._backbone in ['vgg16', 'resnet50']), "{} is not supported".format(self._backbone)
        
        #########change#########
        extract_mode = 'vgg'


        if extract_mode == 'vgg':
            image_size = 224  ##change, orig = 224 for vgg
        else:
            image_size = 384  ##change, for vit
            target_dim=(384, 384)
        ##################

        if feature_type == 'gen_head':
            gen_Head = 1  ####control Head gen. mode
            print('gen_Haed = 1')
        else:
            gen_Head = 0  ####control Head gen. mode

        # print('gen_Haed = ', gen_Head,crop_type)
        # input('...')
        # gen_Head = 1 ####control Head gen. mode
        

        if not gen_Head:
            convnet = backbone_dict[self._backbone](input_shape=(224, 224, 3),
                                                    include_top=False, weights='imagenet') if process else None
        sequences = []

        #####change Head####################################
        if gen_Head:
            MultiP_Head_seq = {}
            SingleP_Head_seq = {}

            #####change for Head model initialize
            c = 0
            fake = 0
            total = 0
            # gen_Head = 0

            parser = argparse.ArgumentParser(description='whenet demo with yolo')
            parser.add_argument('--video', type=str, default='IMG_0176.mp4',         help='path to video file. use camera if no file is given')
            parser.add_argument('--snapshot', type=str, default='/datadisk/PIE/Research/PIEPredict-master/HeadPoseEstimation_WHENet_master/WHENet.h5', help='whenet snapshot path')
            parser.add_argument('--display', type=str, default='simple', help='display all euler angle (simple, full)')
            parser.add_argument('--score', type=float, default=0.3, help='yolo confidence score threshold')
            parser.add_argument('--iou', type=float, default=0.3, help='yolo iou threshold')
            parser.add_argument('--gpu', type=str, default='1', help='gpu')
            parser.add_argument('--output', type=str, default='test.avi', help='output video name')
            
            args = parser.parse_args(args=[])
            # args = parser.parse_args()  ##orig
            # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


            #####turn on for generate head
            # whenet = WHENet(snapshot=args.snapshot)
            # yolo = YOLO(**vars(args))
            print('\n\nGen Head : ',data_type, '...\n\n' )
            
            Head_save_path = save_path[:-9] + "HeadPose_" + data_type + '/' + dataset_F
            # print(save_path[:-9], Head_save_path)
            # if not os.path.exists(Head_save_path):
            #     os.makedirs(Head_save_path)
            print('head_save_path: ', Head_save_path)
            # input('wait...')
            # detector = RetinaFace(model_path='/data1/PIE_2/Resnet50_Final.pth', network="resnet50")
            detector = RetinaFace()

            head_model = SixDRepNet()
            # input('Retina...')
        ###########################################



        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False

                if gen_Head and imp.split('/')[-4] == 'JAAD':  ##change for Gen head mode JAAD because no set
                    set_id = 'noset'
                else:
                    set_id = imp.split('/')[-3]

                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                # Modify the path depending on crop mode
                if crop_type == 'none':
                    img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
                else:
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')


                ##############change for headpose#################
                if gen_Head and crop_type == 'bbox':
                    # print(save_path[:-10]+"HeadPose/HeadPose_"+ set_id + '/' + img_name)
                    # input('wait...')

                    if not os.path.exists(Head_save_path + "/HeadPose_"+ set_id + ".pkl"):   ###if this Headpose not exist
                        # print("#######################")
                        # print("Generating Head")
                        # print("#######################")
                        if 'flip' in imp:
                            imp = imp.replace('_flip', '')
                            flip_image = True

                        img_data = load_img(imp)   ####orig for whenet + yolo
                        img_data = io.imread(imp)    ##change for 6DRepNet + RetinaFace
                        
                        ####change
                            # print("\n\n####")
                            # print(len(img_sequences))
                            # print(type(img_data))
                            # print(img_data.format)
                            # print(img_data.mode)
                            # print(img_data.size)
                            # plt.imshow(img_data)
                            # plt.show()
                            # print("\n\n####")
                            #####
                        
                        if flip_image:
                            # img_data = img_data.transpose(Image.FLIP_LEFT_RIGHT)    ####orig for whenet + yolo
                            img_data = img_data[:, ::-1, :]     ##change for 6DRepNet + RetinaFace
                                              
                        # cropped_image = img_data.crop(list(map(int, b[0:4])))   ####orig for whenet + yolo
                        b = list(map(int, b[0:4]))
                        cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]   ##change for 6DRepNet + RetinaFace
                            
                        img_pil = cropped_image
                        img_ski = cropped_image
                        img_cv = np.array(img_pil)
                         

                        # print(img_pil)
                        # print(np.array(img_pil).shape)

                        
                        bboxes=[]

                        # bboxes, scores, classes = yolo.detect(img_pil)  ####orig for whenet + yolo
                        t3 = time.time()
                        # print('time:', t3)
                        faces = detector(img_ski)    ##change for 6DRepNet + RetinaFace
                        t4 = time.time()

                        # if len(bboxes) == 0:   ####orig for whenet + yolo
                        if len(faces) == 0:
                            # c+=1
                            # total+=1
                            pitch, roll, yaw = 0,0,45
                            # print('no face')
                        else:
                            # total+=1
                            # frame, pitch, roll, yaw = self.process_detection(whenet, img_cv, bboxes[0]) ###take first bbox in bboxes   ####orig for whenet + yolo

                            box_RetiFace = faces[0][0]   ##first [0] for first face in faces list, second[0] for 'box' in faces[0] which format is (box, landmarks, score)
                            x_min = int(box_RetiFace[0])
                            y_min = int(box_RetiFace[1])
                            x_max = int(box_RetiFace[2])
                            y_max = int(box_RetiFace[3])
                            bbox_width = abs(x_max - x_min)
                            bbox_height = abs(y_max - y_min)

                            x_min = max(0, x_min-int(0.1*bbox_height))
                            y_min = max(0, y_min-int(0.1*bbox_width))
                            x_max = x_max+int(0.1*bbox_height)
                            y_max = y_max+int(0.1*bbox_width)

                            img_to_cv = cv2.cvtColor(img_ski, cv2.COLOR_RGB2BGR)
                            img_crop_face = img_to_cv[y_min:y_max, x_min:x_max] 

                            t5 = time.time()
                            pitch, yaw, roll = head_model.predict(img_crop_face)   ##img_to_cv is Because image will be transform to RGB format again in 6DRepNet predict func. (regressor.py line 64)
                                                                          ##change for 6DRepNet + RetinaFace
                            t6 = time.time()
                            print('time1:', t4-t3)
                            print('time2:', t6-t5)
                            print('\ntime:', (t4-t3) + (t6-t5))
                            print("FPS: ", 1/((t4-t3) + (t6-t5)))
                            input('wait...')


                            # head_model.draw_axis(img_crop_face, yaw, pitch, roll)
                            # cv2.imshow("test_window", img_crop_face)
                            # cv2.waitKey(0)
                        # print(type(SingleP_Head_seq))

                        SingleP_Head_seq = self.addtwodimdict(SingleP_Head_seq, set_id, img_name + '_' + p[0], (pitch, roll, yaw))   #insert HeadPose to two dim Dict
                        # print('\n\n')
                        # print(SingleP_Head_seq.keys())    
        # sdfghhikl
                ################### change Head end ####################
                

                ###change fot vit to change the vit pretrained img save path
                if extract_mode == 'vgg':
                    # print('img folder: \n',img_save_folder)
                    pass
                else:
                    # print('\n', save_path[:-1])
                    vit_pre_save_path = '/data1/PIE_2/features/pie/' + feature_type +  '_vit' + '/'
                    # print(vit_pre_save_path)
                    # input('...')
                    # print('\n',vit_pre_save_path)
                    
                    img_save_folder = os.path.join(vit_pre_save_path, set_id, vid_id)   ###orig
                    # print('img folder: \n',img_save_folder)
                    # input('...')

                    if crop_type == 'none':
                        img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
                    else:
                        img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')
                    
                    print('img save path: \n',img_save_path)
                    # input('...')

                ################


                if not gen_Head:
                #     # Check whether the file exists
                    # regen_data = True   ###change for test FPS, must turn off for normal use
                    if os.path.exists(img_save_path) and not regen_data:
                        if not self._generator:
                            with open(img_save_path, 'rb') as fid:
                                try:
                                    img_features = pickle.load(fid)
                                except:
                                    img_features = pickle.load(fid, encoding='bytes')
                            # print("pass",imp)
                            # input('...')
                                
                    else:
                        print('no feature data, need process >>> img path: ',imp)
                        # input('mdf...')
                        if 'flip' in imp:
                            imp = imp.replace('_flip', '')
                            flip_image = True
                        if crop_type == 'none':
                            img_data = cv2.imread(imp)
                            img_features = cv2.resize(img_data, target_dim)
                            if flip_image:
                                img_features = cv2.flip(img_features, 1)
                        else:
                            img_data = cv2.imread(imp)
                            if flip_image:
                                img_data = cv2.flip(img_data, 1)
                            if crop_type == 'bbox':
                                b = list(map(int, b[0:4]))
                                cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                                img_features = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                            elif 'context' in crop_type:
                                bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                                bbox = squarify(bbox, 1, img_data.shape[1])
                                bbox = list(map(int, bbox[0:4]))
                                cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                            elif 'surround' in crop_type:
                                b_org = list(map(int, b[0:4])).copy()
                                bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                                bbox = squarify(bbox, 1, img_data.shape[1])
                                bbox = list(map(int, bbox[0:4]))
                                img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                                cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                            else:
                                raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                        if preprocess_input is not None:
                            ###change for vit
                            if extract_mode == 'vgg':
                                img_features = preprocess_input(img_features)   #orig only this line
                            else:
                            ###change for vit
                                image = utils.read(img_features, image_size)
                                preprocessed_img = vit.preprocess_inputs(image)   #####orig :    .reshape(1, image_size, image_size, 3)
                            ###########
                        if process:
                            expanded_img = np.expand_dims(img_features, axis=0)

                            ######test FPS#########
                            tVGG1 = time.time()

                            img_features = convnet.predict(expanded_img)  ##orig

                            tVGG2 = time.time()
                            print('time:', tVGG2-tVGG1)
                            print('VGG FPS:', 1/(tVGG2-tVGG1))
                            input('wait VGG FPS...')
                            ######################

                        # Save the file
                        if not os.path.exists(img_save_folder):
                            os.makedirs(img_save_folder)
                        with open(img_save_path, 'wb') as fid:
                            pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                    if process and not self._generator:
                        if extract_mode == 'vgg':
                        #####orig for vgg
                            if self._global_pooling == 'max':
                                img_features = np.squeeze(img_features)
                                img_features = np.amax(img_features, axis=0)
                                img_features = np.amax(img_features, axis=0)
                            elif self._global_pooling == 'avg':
                                img_features = np.squeeze(img_features)
                                img_features = np.average(img_features, axis=0)
                                img_features = np.average(img_features, axis=0)
                            else:
                                img_features = img_features.ravel()
                        else:
                            img_features = np.squeeze(img_features)

                    # if using the generator save the cached features path and size of the features                                   
                    if self._generator:
                        img_seq.append(img_save_path)
                    else:
                        img_seq.append(img_features)
            if not gen_Head:
                sequences.append(img_seq)
        if not gen_Head:
            sequences = np.array(sequences)

        #########change for save HeadPose############################
        # if data_type=='train':
        
        if gen_Head and crop_type == 'bbox':
            print("Dict:")
            print(SingleP_Head_seq) 

            if not os.path.exists(Head_save_path):
                os.makedirs(Head_save_path)
            for sid in SingleP_Head_seq.keys():
                with open(Head_save_path + "/HeadPose_" + sid + ".pkl", 'wb') as HP:
                    print("Head pkl save path:")
                    print(Head_save_path + "/HeadPose_" + sid + ".pkl")
                    pickle.dump(SingleP_Head_seq[sid], HP, pickle.HIGHEST_PROTOCOL)
        ################################################

        # input('Head Gen* Finish!!')
        


        # compute size of the features after the processing
        if not gen_Head:        
            if self._generator:
                with open(sequences[0][0], 'rb') as fid:
                    feat_shape = pickle.load(fid).shape
                if process:
                    if self._global_pooling in ['max', 'avg']:
                        feat_shape = feat_shape[-1]
                    else:
                        feat_shape = np.prod(feat_shape)
                if not isinstance(feat_shape, tuple):
                    feat_shape = (feat_shape,)
                feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
            else:
                feat_shape = sequences.shape[1:]

        ########## Head ###########
        if gen_Head:
            sequences = []   ######$$$$ remember to trun off $$$$####################
            feat_shape = []   ######$$$$ remember to trun off $$$$####################

        return sequences, feat_shape

        # Processing images anf generate features

    def get_optical_flow(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
               \nsave_path={}, ".format(data_type, crop_type, crop_mode, save_path))
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        # flow size (h,w)
        flow_size = read_flow_file(img_sequences[0][0].replace('images', 'optical_flow').replace('png', 'flo')).shape
        img_size = cv2.imread(img_sequences[0][0]).shape
        # A ratio to adjust the dimension of bounding boxes (w,h)
        box_resize_coef = (flow_size[1]/img_size[1], flow_size[0]/img_size[0])

        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            flow_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                optflow_save_folder = os.path.join(save_path, set_id, vid_id)
                ofp = imp.replace('images', 'optical_flow').replace('png', 'flo')
                # Modify the path depending on crop mode
                if crop_type == 'none':
                    optflow_save_path = os.path.join(optflow_save_folder, img_name + '.flo')
                else:
                    optflow_save_path = os.path.join(optflow_save_folder, img_name + '_' + p[0] + '.flo')

                # Check whether the file exists
                if os.path.exists(optflow_save_path) and not regen_data:
                    if not self._generator:
                        ofp_data = read_flow_file(optflow_save_path)
                else:
                    if 'flip' in imp:
                        ofp = ofp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        ofp_image = read_flow_file(ofp)
                        ofp_data = cv2.resize(ofp_image, target_dim)
                        if flip_image:
                            ofp_data = cv2.flip(ofp_data, 1)
                    else:
                        ofp_image = read_flow_file(ofp)
                        # Adjust the size of bbox according to the dimensions of flow map
                        b = list(map(int, [b[0] * box_resize_coef[0], b[1] * box_resize_coef[1],
                                           b[2] * box_resize_coef[0], b[3] * box_resize_coef[1]]))
                        if flip_image:
                            ofp_image = cv2.flip(ofp_image, 1)
                        if crop_type == 'bbox':
                            cropped_image = ofp_image[b[1]:b[3], b[0]:b[2], :]
                            ofp_data = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, ofp_image.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = ofp_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            ofp_data = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        elif 'surround' in crop_type:
                            b_org = b.copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, ofp_image.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            ofp_image[b_org[1]:b_org[3], b_org[0]: b_org[2], :] = 0
                            cropped_image = ofp_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            ofp_data = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))

                    # Save the file
                    if not os.path.exists(optflow_save_folder):
                        os.makedirs(optflow_save_folder)
                    write_flow(ofp_data, optflow_save_path)

                # if using the generator save the cached features path and size of the features
                if self._generator:
                    flow_seq.append(optflow_save_path)
                else:
                    flow_seq.append(ofp_data)
            sequences.append(flow_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            feat_shape = read_flow_file(sequences[0][0]).shape
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]
        return sequences, feat_shape

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            opts:  Options for generating data samples
        Returns:
            A list of data samples extracted from raw data
            Positive and negative data counts
        """
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')

        ##################################

        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()

        ####################################

        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
        else:
            overlap = opts['overlap'] # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:   ##for every pid(sample)
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs

            for seq in data_raw['bbox']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_context_data(self, model_opts, data, data_type, feature_type):
        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')
        process = model_opts.get('process', True)
        aux_name = [self._backbone]
        if not process:
            aux_name.append('raw')
        aux_name = '_'.join(aux_name).strip('_')
        eratio = model_opts['enlarge_ratio']
        dataset = model_opts['dataset']

        data_gen_params = {'data_type': data_type, 'crop_type': 'none',
                           'target_dim': model_opts.get('target_dim', (224, 224))}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
            data_gen_params['dataset_F'] = model_opts['dataset_F']   ##change
        elif 'local_context' in feature_type:
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
            data_gen_params['dataset_F'] = model_opts['dataset_F']   ##change
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
            data_gen_params['dataset_F'] = model_opts['dataset_F']   ##change
        elif 'scene_context' in feature_type:
            data_gen_params['crop_type'] = 'none'
            data_gen_params['dataset_F'] = model_opts['dataset_F']   ##change
        elif 'gen_head' in feature_type:  ###change for special gen head mode
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['dataset_F'] = model_opts['dataset_F']   ##change
        save_folder_name = feature_type
        # if 'flow' not in feature_type:
            # save_folder_name = '_'.join([feature_type, aux_name]) ##change
            # if 'local_context' in feature_type or 'surround' in feature_type:
            #     save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'], _ = get_path(save_folder=save_folder_name,
                                                   dataset=dataset, save_root_folder='/mnt/sda/PIE/Research/PIEPredict-master/data/features')  ###change
                                                                                    ##orig is "/datadisk/PIE/Research/PIEPredict-master/data/features" before rebuild server __2024.5.14
        if 'flow' in feature_type:                                                          
            return self.get_optical_flow(data['image'],
                                         data['box_org'],
                                         data['ped_id'],
                                         **data_gen_params)
        else:
            return self.load_images_crop_and_process(data['image'],
                                                     data['box_org'],
                                                     data['ped_id'],
                                                     feature_type,
                                                     process=process,
                                                     **data_gen_params)
                                                    
    def get_Head(self, img_sequences,
                      ped_ids, file_path,
                      dataset,
                      data_type='train'):
        """
        Reads the pie HeadPose from saved .pkl files
        :param img_sequences: Sequences of image names
        :param ped_ids: Sequences of pedestrian ids
        :param file_path: Path to where poses are saved
        :param data_type: Whether it is for training or testing
        :return: Sequences of poses
        """

        print('\n#####################################')
        print('Getting HeadPose %s' % data_type)
        print('#####################################')
        Head_all = []
        set_Head_list = os.listdir(file_path)
        set_Head = {}

        # print('\nset list#####################################')
        print(file_path)
        # print('#####################################')
        
        ###orig
        for s in set_Head_list:
            with open(os.path.join(file_path, s), 'rb') as fid:
                try:
                    p = pickle.load(fid)
                except:
                    p = pickle.load(fid, encoding='bytes')
            set_Head[s.split('.pkl')[0].split('_')[-1]] = p
        ###

        # print(os.path.join(file_path, s))
        # print(img_sequences)
        # input("wait..")
            
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            Head = []

            for imp, p in zip(seq, pid): 

                if imp.split('/')[-4] == 'JAAD':  ##change for JAAD because no set
                    set_id = 'noset'
                else:
                    set_id = imp.split('/')[-3]

                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]  ##
                k = img_name + '_' + p[0]
                
                # print("###this is K")
                # print(k)
                # print('#####################################')
                
                # print(set_Head[set_id].keys())
                # print(k)
                
                ###orig
                if k in set_Head[set_id].keys():
                    # [pitch ,roll,yaw]
                    ##orig
                    Head.append(np.squeeze(set_Head[set_id][k]))
                    # print(set_Head[set_id][k])
                    # input("wait...")
                    ##change for abs Head angle
                    # Head.append(list(map(abs, set_Head[set_id][k])))
                else:
                    Head.append([0, 0, 45])
                ###                   
            Head_all.append(Head)
        Head_all = np.array(Head_all)
        # print('this head:', (Head_all), len(Head_all), type(Head_all))
        # input('...')
        return Head_all

    ####chang to get traffic light
    def get_TL(self, img_sequences,
                      ped_ids, file_path,
                      dataset,
                      data_type='train'):
        """
        Reads the pie HeadPose from saved .pkl files
        :param img_sequences: Sequences of image names
        :param ped_ids: Sequences of pedestrian ids
        :param file_path: Path to where poses are saved
        :param data_type: Whether it is for training or testing
        :return: Sequences of poses
        """

        print('\n#####################################')
        print('Getting Traffic light %s' % data_type)
        print('#####################################')
        TL_all = []
        set_Head_list = os.listdir(file_path)
        set_Head = {}

        # print('\nset list#####################################')
        # print(file_path)
        # print('#####################################')
        
        ###orig
        if 'jaad_beh' in dataset:
            path_TL = '/data1/PIE_2/JAADbeh_Traffic_Light_data.pkl'
        elif 'jaad_all' in dataset:
            path_TL = '/data1/PIE_2/JAADall_Traffic_Light_data.pkl'
        else:
            path_TL = '/data1/PIE_2/Traffic_Light_data.pkl'


        with open(path_TL, 'rb') as fid:
            try:
                TL = pickle.load(fid)
            except:
                TL = pickle.load(fid, encoding='bytes')

        ###
        # print(img_sequences)
            
        i = -1
        for seq in (img_sequences):
            i += 1
            update_progress(i / len(img_sequences))
            TL_PerFrame = []

            for imp in (seq): 
                # set_id = imp.split('/')[-3]
                if imp.split('/')[-4] == 'JAAD':  ##change for JAAD because no set
                    set_id = 'jaad_no_set'
                else:
                    set_id = imp.split('/')[-3]   ##pie turn on

                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]  ##
                # k = img_name + '_' + p[0]
                
                # print("###this is K")
                # print(k)
                # print('#####################################')
                
       
                if img_name[-5:] == '_flip':
                    img_name = img_name[:-5]


                if int(img_name) in TL[set_id][vid_id].keys():
                    # [pitch ,roll,yaw]
                    TL_PerFrame.append([list(TL[set_id][vid_id][int(img_name)].values())[0][0]]) ###transfer dict_values to list, first [0] is for redundant outer list 
                else:
                    TL_PerFrame.append([0])    
                ###                   
            # input('Get TL 1038')
            TL_all.append(TL_PerFrame)


        TL_all = np.array(TL_all)
        # print(TL_all)
        # print(len(TL_all))
        # print(TL_all[145])
        # print(len(TL_all[145]))
        # input('Get TL 1046')

        return TL_all

    ####chang to get traffic light one hot
    def get_TL_OneHot(self, img_sequences,
                      ped_ids, file_path,
                      dataset,
                      data_type='train'):
        """
        Reads the pie HeadPose from saved .pkl files
        :param img_sequences: Sequences of image names
        :param ped_ids: Sequences of pedestrian ids
        :param file_path: Path to where poses are saved
        :param data_type: Whether it is for training or testing
        :return: Sequences of poses
        """

        print('\n#####################################')
        print('Getting Traffic light %s' % data_type)
        print('#####################################')
        TL_all = []
        set_Head_list = os.listdir(file_path)
        set_Head = {}

        # print('\nset list#####################################')
        # print(file_path)
        # print('#####################################')
        
        ###orig

        if 'jaad_beh' in dataset:
            path_TL = '/data1/PIE_2/JAADbeh_Traffic_Light_data.pkl'
        elif 'jaad_all' in dataset:
            path_TL = '/data1/PIE_2/JAADall_Traffic_Light_data.pkl'
        else:
            path_TL = '/data1/PIE_2/Traffic_Light_data.pkl'


        with open(path_TL, 'rb') as fid:
            try:
                TL = pickle.load(fid)
            except:
                TL = pickle.load(fid, encoding='bytes')

        ###
        # print(img_sequences)
            
        i = -1
        for seq in (img_sequences):
            i += 1
            update_progress(i / len(img_sequences))
            TL_PerFrame = []

            for imp in (seq): 
                # set_id = imp.split('/')[-3]
                if imp.split('/')[-4] == 'JAAD':  ##change for JAAD because no set
                    set_id = 'jaad_no_set'
                else:
                    set_id = imp.split('/')[-3]   ##pie turn on


                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]  ##
                # k = img_name + '_' + p[0]
                
                # print("###this is K")
                # print(k)
                # print('#####################################')
                
       
                if img_name[-5:] == '_flip':
                    img_name = img_name[:-5]


                if int(img_name) in TL[set_id][vid_id].keys():
                    # [pitch ,roll,yaw]
                    TL_PerFrame.append([list(TL[set_id][vid_id][int(img_name)].values())[0][0]]) ###transfer dict_values to list, first [0] is for redundant outer list 

                else:
                    TL_PerFrame.append([0])    
                ###                   
            TL_all.append(TL_PerFrame)
        TL_all = np.array(TL_all)
        # print(TL_all)
        # print(len(TL_all))
        # print(TL_all[145])
        # print(len(TL_all[145]))

        return TL_all

    ####chang to get Signs
    def get_Sign(self, img_sequences,
                      ped_ids, file_path,
                      dataset,
                      data_type='train'):
        """
        Reads the pie HeadPose from saved .pkl files
        :param img_sequences: Sequences of image names
        :param ped_ids: Sequences of pedestrian ids
        :param file_path: Path to where poses are saved
        :param data_type: Whether it is for training or testing
        :return: Sequences of poses
        """

        print('\n#####################################')
        print('Getting Sign %s' % data_type)
        print('#####################################')
        Sign_all = []
        set_Head_list = os.listdir(file_path)
        set_Head = {}

        # print('\nset list#####################################')
        # print(file_path)
        # print('#####################################')
        
        ###orig

        if 'jaad_beh' in dataset:
            path_TL = '/data1/PIE_2/JAADbeh_Signs_data_New.pkl'
        elif 'jaad_all' in dataset:
            path_TL = '/data1/PIE_2/JAADall_Signs_data_New.pkl'
        else:
            path_TL = '/data1/PIE_2/Signs_data.pkl'    #####!@!!!need to be check!!!


        with open(path_TL, 'rb') as fid:
            try:
                Sign = pickle.load(fid)
            except:
                Sign = pickle.load(fid, encoding='bytes')
        
        print(dataset)
        print(Sign.keys())
        # asdf
        ###
        # print(img_sequences)
            
        i = -1
        for seq in (img_sequences):
            i += 1
            update_progress(i / len(img_sequences))
            Sign_PerFrame = []

            for imp in (seq): 
                # set_id = imp.split('/')[-3]   ###orig

                if imp.split('/')[-4] == 'JAAD':  ##change for JAAD because no set
                    set_id = 'jaad_no_set'
                else:
                    set_id = imp.split('/')[-3]   ##pie turn on

                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]  ##
                # k = img_name + '_' + p[0]
                
                # print("###this is K")
                # print(k)
                # print('#####################################')
                
       
                if img_name[-5:] == '_flip':
                    img_name = img_name[:-5]


                if int(img_name) in Sign[set_id][vid_id].keys():
                    # [pitch ,roll,yaw]
                    # Sign_PerFrame.append([list(Sign[set_id][vid_id][int(img_name)].values())[0][0]]) ###transfer dict_values to list, first [0] is for redundant outer list ###this is for append orignal sign class index
                    if 1 <= list(Sign[set_id][vid_id][int(img_name)].values())[0][0] <= 5:
                        # print('\n', list(Sign[set_id][vid_id][int(img_name)].values())[0][0])
                        Sign_PerFrame.append([5]) ###this is for : if sign class between 1 to 5, append 1
                        # Sign_PerFrame.append([1]) ###this is for : if sign class between 1 to 5, append 1
                    else:
                        Sign_PerFrame.append([0]) 
                else:
                    Sign_PerFrame.append([0])    
                ###                   
            Sign_all.append(Sign_PerFrame)
        Sign_all = np.array(Sign_all)
        # for i in Sign_all:
        #     print(i)
        
        # print(len(Sign_all))
        # print(Sign_all[145])
    
        # print(len(Sign_all[145]))
        # print(Sign_all.shape)
        # input("Press Enter to continue...")
        
        return Sign_all

    ####chang to get Signs one hot
    def get_Sign_OneHot(self, img_sequences,
                      ped_ids, file_path,
                      dataset,
                      data_type='train'):
        """
        Reads the pie HeadPose from saved .pkl files
        :param img_sequences: Sequences of image names
        :param ped_ids: Sequences of pedestrian ids
        :param file_path: Path to where poses are saved
        :param data_type: Whether it is for training or testing
        :return: Sequences of poses
        """

        print('\n#####################################')
        print('Getting Sign %s' % data_type)
        print('#####################################')
        Sign_all = []
        set_Head_list = os.listdir(file_path)
        set_Head = {}

        # print('\nset list#####################################')
        # print(file_path)
        # print('#####################################')
        
        ###orig
        if 'jaad_beh' in dataset:
            path_TL = '/data1/PIE_2/JAADbeh_Signs_data_New.pkl'
        elif 'jaad_all' in dataset:
            path_TL = '/data1/PIE_2/JAADall_Signs_data_New.pkl'
        else:
            path_TL = '/data1/PIE_2/Signs_data_New.pkl'


        with open(path_TL, 'rb') as fid:
            try:
                Sign = pickle.load(fid)
            except:
                Sign = pickle.load(fid, encoding='bytes')

        ###
        # print(img_sequences)
        # input("img sequence...")

        i = -1
        for seq in (img_sequences):
            i += 1
            update_progress(i / len(img_sequences))
            Sign_PerFrame = []

            for imp in (seq): 
                # set_id = imp.split('/')[-3]
                if imp.split('/')[-4] == 'JAAD':  ##change for JAAD because no set
                    set_id = 'jaad_no_set'
                else:
                    set_id = imp.split('/')[-3]   ##pie turn on

                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]  ##
                # k = img_name + '_' + p[0]
                
                # print("###this is K")
                # print(k)
                # print('#####################################')
                
       
                if img_name[-5:] == '_flip':
                    img_name = img_name[:-5]


                if int(img_name) in Sign[set_id][vid_id].keys():
                    # [pitch ,roll,yaw]
                    Sign_PerFrame.append([list(Sign[set_id][vid_id][int(img_name)].values())[0][0]]) ###transfer dict_values to list, first [0] is for redundant outer list ###this is for append orignal sign class index
                    # for aa, S_type in Sign[set_id][vid_id][int(img_name)].items():
                    #     print('\n', img_name, aa, S_type[0])
                        # input("Press Enter to continue...")
                    # S_type = [S[0] for S in Sign[set_id][vid_id][int(img_name)].values()]
                    # Sign_PerFrame.append(S_type)
                    # print([S[0] for S in Sign[set_id][vid_id][int(img_name)].values()])
                    # input("Press Enter to continue...")
                    

                    # if 1 <= list(Sign[set_id][vid_id][int(img_name)].values())[0][0] <= 5:
                    #     # print('\n', list(Sign[set_id][vid_id][int(img_name)].values())[0][0])
                    #     Sign_PerFrame.append([5]) ###this is for : if sign class between 1 to 5, append 1
                        ## Sign_PerFrame.append([1]) ###this is for : if sign class between 1 to 5, append 1
                    # else:
                    #     Sign_PerFrame.append([0]) 
                else:
                    Sign_PerFrame.append([0])    
                ###                   
            Sign_all.append(Sign_PerFrame)
        Sign_all = np.array(Sign_all)
        # for i in Sign_all:
        #     print(i)
        
        # print(len(Sign_all))
        # print(Sign_all[145])
    
        # print(len(Sign_all[145]))
        # print(Sign_all.shape)
        # input("Press Enter to continue...")
        
        return Sign_all

    ####chang to get Crosswalk
    def get_CW(self, img_sequences,
                      ped_ids, file_path,
                      dataset,
                      data_type='train'):
        """
        Reads the pie HeadPose from saved .pkl files
        :param img_sequences: Sequences of image names
        :param ped_ids: Sequences of pedestrian ids
        :param file_path: Path to where poses are saved
        :param data_type: Whether it is for training or testing
        :return: Sequences of poses
        """

        print('\n#####################################')
        print('Getting Crosswalk %s' % data_type)
        print('#####################################')
        Crosswalk_all = []
        set_Head_list = os.listdir(file_path)
        set_Head = {}

        # print('\nset list#####################################')
        # print(file_path)
        # print('#####################################')
        
        ###orig
        if 'jaad_beh' in dataset:
            path_TL = '/data1/PIE_2/JAADbeh_Crosswalk_data.pkl'
        elif 'jaad_all' in dataset:
            path_TL = '/data1/PIE_2/JAADall_Crosswalk_data.pkl'
        else:
            path_TL = '/data1/PIE_2/Crosswalk_data.pkl'


        with open(path_TL, 'rb') as fid:
            try:
                Crosswalk = pickle.load(fid)
            except:
                Crosswalk = pickle.load(fid, encoding='bytes')

        ###
        # print(img_sequences)
            
        i = -1
        for seq in (img_sequences):
            i += 1
            update_progress(i / len(img_sequences))
            Crosswalk_PerFrame = []

            for imp in (seq): 
                # set_id = imp.split('/')[-3]
                if imp.split('/')[-4] == 'JAAD':  ##change for JAAD because no set
                    set_id = 'jaad_no_set'
                else:
                    set_id = imp.split('/')[-3]   ##pie turn on

                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]  ##
                # k = img_name + '_' + p[0]
                
                # print("###this is K")
                # print(k)
                # print('#####################################')
                
       
                if img_name[-5:] == '_flip':
                    img_name = img_name[:-5]


                if int(img_name) in Crosswalk[set_id][vid_id].keys():
                    # [pitch ,roll,yaw]
                    Crosswalk_PerFrame.append([list(Crosswalk[set_id][vid_id][int(img_name)].values())[0][0]]) ###transfer dict_values to list, first [0] is for redundant outer list   ##chang for thesis demo
                    # Crosswalk_PerFrame.append([5* list(Crosswalk[set_id][vid_id][int(img_name)].values())[0][0]]) ###transfer dict_values to list, first [0] is for redundant outer list 
                    # print(Crosswalk_PerFrame)
                    # input("Press Enter to continue...")                                             
                else:
                    Crosswalk_PerFrame.append([0])    
                ###                   
            Crosswalk_all.append(Crosswalk_PerFrame)
        Crosswalk_all = np.array(Crosswalk_all)
        # print(Crosswalk_all)
        # print(len(Crosswalk_all))
        # print(Crosswalk_all[145])
        # print(len(Crosswalk_all[145]))

        return Crosswalk_all

    #### def get_pose now move to utils.py #####

    def get_data(self, data_type, data_raw, model_opts):
        """
        Generates data train/test/val data
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            model_opts: Model options for generating data
        Returns:
            A dictionary containing, data, data parameters used for model generation,
            effective dimension of data (the number of rgb images to be used calculated accorfing
            to the length of optical flow window) and negative and positive sample counts
        """

        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data_F = model_opts['dataset_F']
        
        ###This is where every ped sample be split into a different part as an individual smaple 
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        # print(data)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
                # features_test, feat_shape_test = self.get_context_data(model_opts, data, 'test', d_type)
                # features_val, feat_shape_val = self.get_context_data(model_opts, data, 'val', d_type)
                # asdasd
            elif 'gen_head' in d_type:
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
                features_test, feat_shape_test = self.get_context_data(model_opts, data, 'test', d_type)
                features_val, feat_shape_val = self.get_context_data(model_opts, data, 'val', d_type)
                asdasd

            elif 'pose' == d_type:
                threeD = 1 ##change
                if threeD != 1:  ####orig 2D pose
                    path_to_pose, _ = get_path(save_folder='poses',
                                            dataset=dataset,
                                            save_root_folder='../data/features')
                else:   ###3Dpose
                    path_to_pose, _ = get_path(save_folder='poses3D',
                                            dataset=dataset,
                                            save_root_folder='../data/features')
                                                                                       
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'],
                                    threeD = threeD) ##change
                feat_shape = features.shape[1:]

            ####change
            # HeadPose features
            elif 'headpose' == d_type:
                print('\n#####################################')
                print('Generating headpose %s' % data_type)
                print('#####################################')

                if data_F == 'pie':    ###change for jaad_all & jaad_beh
                    head_path = 'HeadPose_' + data_type
                else:
                    head_path = 'HeadPose_' + data_type + '/' + data_F
                print('data_F = ', data_F , 'head_path = ', head_path)

                path_to_headpose, _ = get_path(save_folder=head_path,   ###change for jaad_all & jaad_beh
                                                      dataset=dataset,
                                                      save_root_folder='../data/features')

                # print(path_to_headpose)
                # input("wait..")

                features = self.get_Head(data['image'],
                                           data['ped_id'], data_type = data_type,
                                           dataset= data_F,
                                           file_path=path_to_headpose)
                feat_shape = features.shape[1:]
                
            #####
            ####change
            # Traffic light features
            elif 'traffic_light' in d_type:
                print('\n#####################################')
                print('Generating traffic light %s' % data_type)
                print('#####################################')

                ##unnecessary, to be remove (todo) 
                path_to_headpose, _ = get_path(save_folder='HeadPose',
                                                      dataset=dataset,
                                                      save_root_folder='data/features')
                ##

                tl_temp = self.get_TL(data['image'],
                                           data['ped_id'], data_type=data_type,
                                           dataset= data_F,
                                           file_path=path_to_headpose)

                tl_new = np.zeros((tl_temp.shape[0], tl_temp.shape[1], 3))

                # print(type(tl_temp))
                for id_s, s in enumerate(tl_temp):
                    for id_f, f in enumerate(s):
                        # print(tl_temp[id_s][id_f])
                        if f == 1:
                            # tl_new[id_s][id_f] = [5, 0, 0]
                            tl_new[id_s][id_f] = [1]   #for thesis demo no concate
                        elif f == 2:
                            # tl_new[id_s][id_f] = [0, 5, 0]
                            tl_new[id_s][id_f] = [2]   #for thesis demo no concate
                        elif f == 3:
                            # tl_new[id_s][id_f] = [0, 0, 5]
                            tl_new[id_s][id_f] = [3]    #for thesis demo no concate
                        else:
                            # tl_new[id_s][id_f] = [0, 0, 0]
                            tl_new[id_s][id_f] = [0]    #for thesis demo no concate

                # input('get data 1898')

                # features = tl_new
                #####################
                # f_sign = self.get_Sign_OneHot(data['image'],      ##change for thesis
                #                             data['ped_id'], data_type=data_type,
                #                             dataset= data_F,
                #                             file_path=path_to_headpose)

                # f_CW = self.get_CW(data['image'],
                #                            data['ped_id'], data_type=data_type,
                #                            dataset= data_F,
                #                            file_path=path_to_headpose)

                # features = np.concatenate((tl_temp, f_sign, f_CW), axis= -1)
                ##########################

                features = tl_temp


                feat_shape = features.shape[1:]
                # print(data[k]['traffic_light'])
                # print(data_type_sizes_dict['traffic_light'])      
                # 
                          
            # Sign features
            elif 'sign' in d_type:
                print('\n#####################################')
                print('Generating Sign %s' % data_type)
                print('#####################################')

                ##unnecessary, to be remove (todo) 
                path_to_headpose, _ = get_path(save_folder='HeadPose',
                                                      dataset=dataset,
                                                      save_root_folder='data/features')
                ##

                # features = self.get_Sign(data['image'],
                #                             data['ped_id'], data_type=data_type,
                #                             dataset= data_F,
                #                             file_path=path_to_headpose)

                features = self.get_Sign_OneHot(data['image'],      ##change for thesis
                                            data['ped_id'], data_type=data_type,
                                            dataset= data_F,
                                            file_path=path_to_headpose)

                feat_shape = features.shape[1:]
                # print(data[k]['Sign'])
                # print(data_type_sizes_dict['Sign']) 

            # Crosswalk features
            elif 'crosswalk' in d_type:
                print('\n#####################################')
                print('Generating Crosswalk %s' % data_type)
                print('#####################################')

                ##unnecessary, to be remove (todo) 
                path_to_headpose, _ = get_path(save_folder='HeadPose',
                                                      dataset=dataset,
                                                      save_root_folder='data/features')
                ##

                features = self.get_CW(data['image'],
                                           data['ped_id'], data_type=data_type,
                                           dataset= data_F,
                                           file_path=path_to_headpose)
                feat_shape = features.shape[1:]
                # print(data[k]['Crosswalk'])
                # print(data_type_sizes_dict['Crosswalk']) 

            ####change############
            ###TF + sign + CW (sim one-hot)
            elif 'TSC' in d_type:
                print('\n#####################################')
                print('Generating TSC %s' % data_type)
                print('#####################################')

                Sign_OneH = 0

                ##unnecessary, to be remove (todo) 
                path_to_headpose, _ = get_path(save_folder='HeadPose',
                                                      dataset=dataset,
                                                      save_root_folder='data/features')
                ##
                
                if not Sign_OneH:
                    ##orig
                    sign_temp = self.get_Sign(data['image'],
                                            data['ped_id'], data_type=data_type,
                                            dataset= data_F,
                                            file_path=path_to_headpose)
                else:
                    ###change for onehot sign, orig data format: if sign==5 no sign ==0.  
                    sign_temp = self.get_Sign_OneHot(data['image'],
                                            data['ped_id'], data_type=data_type,
                                            dataset= data_F,
                                            file_path=path_to_headpose)
                    
                    # sign_new = np.zeros((sign_temp.shape[0], sign_temp.shape[1], 8))
                    # sign_new = np.zeros((sign_temp.shape[0], sign_temp.shape[1], 6))
                    sign_new = np.zeros((sign_temp.shape[0], sign_temp.shape[1], 4))
                    
                    # for id_s, s in enumerate(sign_temp):
                    #     for id_f, f in enumerate(s):
                    #         # print(tl_temp[id_s][id_f])
                    #         # if len(f)>1:
                    #         #     print(f,f[0])
                    #         #     input('ff...')
                    #         if f[0] == 1:
                    #             sign_new[id_s][id_f] = [1, 0, 0, 0, 0, 0, 0, 0]
                    #         elif f[0] == 2:
                    #             sign_new[id_s][id_f] = [0, 1, 0, 0, 0, 0, 0, 0]
                    #         elif f[0] == 3:
                    #             sign_new[id_s][id_f] = [0, 0, 1, 0, 0, 0, 0, 0]
                    #         elif f[0] == 4:
                    #             sign_new[id_s][id_f] = [0, 0, 0, 1, 0, 0, 0, 0]
                    #         elif f[0] == 5:
                    #             sign_new[id_s][id_f] = [0, 0, 0, 0, 1, 0, 0, 0]
                    #         elif f[0] == 6:
                    #             sign_new[id_s][id_f] = [0, 0, 0, 0, 0, 1, 0, 0]
                    #         elif f[0] == 7:
                    #             sign_new[id_s][id_f] = [0, 0, 0, 0, 0, 0, 1, 0]
                    #         elif f[0] == 8:
                    #             sign_new[id_s][id_f] = [0, 0, 0, 0, 0, 0, 0, 1]
                    #         else:
                    #             sign_new[id_s][id_f] = [0, 0, 0, 0, 0, 0, 0, 0]

                    for id_p, p in enumerate(sign_temp):
                        for id_f, f in enumerate(p):
                            # print(tl_temp[id_s][id_f])
                            # if len(f)>1:
                            #     print(f,f[0])
                            #     input('ff...')

                            # print(f[0])

                            # if 1 <= f[0] <= 4:
                            #     sign_new[id_p][id_f] = [1, 0, 0, 0, 0, 0]
                            # elif f[0] == 5:
                            #     sign_new[id_p][id_f] = [0, 1, 0, 0, 0, 0]
                            # elif f[0] == 6:
                            #     sign_new[id_p][id_f] = [0, 0, 1, 0, 0, 0]
                            # elif f[0] == 7:
                            #     sign_new[id_p][id_f] = [0, 0, 0, 1, 0, 0]
                            # elif f[0] == 8:
                            #     sign_new[id_p][id_f] = [0, 0, 0, 0, 1, 0]
                            # elif f[0] == 9:
                            #     sign_new[id_p][id_f] = [0, 0, 0, 0, 0, 1]
                            # else:
                            #     sign_new[id_p][id_f] = [0, 0, 0, 0, 0, 0]

                            if 1 <= f[0] <= 4:
                                sign_new[id_p][id_f] = [5, 0, 0, 0]
                            elif f[0] == 5:
                                sign_new[id_p][id_f] = [0, 5, 0, 0]
                            elif f[0] == 6:
                                sign_new[id_p][id_f] = [0, 0, 5, 0]
                            elif f[0] == 7:
                                sign_new[id_p][id_f] = [0, 0, 0, 5]
                            else:
                                sign_new[id_p][id_f] = [0, 0, 0, 0]
                        
                        # si = data['sign'].shape[1:]

                if Sign_OneH:
                    sum_dict = {}
                    for i in range(0,len(sign_new[0][0]) + 1):
                        sum_dict[i] = 0

                if Sign_OneH:
                    for pid, _ in enumerate(sign_new):
                        for vid, _ in enumerate(sign_new[pid]):
                            flag = 0
                            for ind, s_type in enumerate(sign_new[pid][vid]):
                                if s_type == 1:
                                    sum_dict[ind + 1] = sum_dict[ind + 1] + 1
                                    flag = 1
                            if flag == 0:
                                sum_dict[0] += 1
                                

                    print(sum_dict)

    

                cw_temp = self.get_CW(data['image'],
                                           data['ped_id'], data_type=data_type,
                                           dataset= data_F,
                                           file_path=path_to_headpose)
                # cw_temp = data['crosswalk'].shape[1:]

                tl_temp = self.get_TL(data['image'],
                                           data['ped_id'], data_type=data_type,
                                           dataset= data_F,
                                           file_path=path_to_headpose)
                # data_type_sizes_dict['traffic_light'] = data['traffic_light'].shape[1:]
                # input('1874')

                tl_new = np.zeros((tl_temp.shape[0], tl_temp.shape[1], 3))

                # print(type(tl_temp))
                for id_s, s in enumerate(tl_temp):
                    for id_f, f in enumerate(s):
                        # print(tl_temp[id_s][id_f])
                        if f == 1:
                            tl_new[id_s][id_f] = [5, 0, 0]
                        elif f == 2:
                            tl_new[id_s][id_f] = [0, 5, 0]
                        elif f == 3:
                            tl_new[id_s][id_f] = [0, 0, 5]
                        else:
                            tl_new[id_s][id_f] = [0, 0, 0]
                # input('get data 1898')
                # print(tl_new)
                # print(tl_new.shape)
                
                ##orig: sign_temp , change for one hot, use orig to get sign num
                if Sign_OneH:
                    sign_temp = sign_new
                
                features = np.concatenate((tl_new, sign_temp, cw_temp), axis= -1)
                # features = np.concatenate((tl_new, cw_temp), axis= -1)

                # features = np.concatenate((tl_new), axis= -1) #change temp
                # features = tl_new

                # input('get data 1905')

                feat_shape = features.shape[1:]
                # print(sign_temp)
                # print(cw_temp)
                # print(data['TSC'].shape)
                # for i in data['TSC']:
                #     print(i)
                #     input('data...')
                # asd

                # print(data['Crosswalk'])
                # print(data_type_sizes_dict['Crosswalk']) 
            
            #############################


            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        print(data_sizes)
        # asdffda

        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'val'), data['crossing']) # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def log_configs(self, config_path, batch_size, epochs,
                    lr, model_opts):

        # TODO: Update config by adding network attributes
        """
        Logs the parameters of the model and training
        Args:
            config_path: The path to save the file
            batch_size: Batch size of training
            epochs: Number of epochs for training
            lr: Learning rate of training
            model_opts: Data generation parameters (see get_data)
        """
        # Save config and training param files
        with open(config_path, 'wt') as fid:
            yaml.dump({'model_opts': model_opts, 
                       'train_opts': {'batch_size':batch_size, 'epochs': epochs, 'lr': lr}},
                       fid, default_flow_style=False)
        # with open(config_path, 'wt') as fid:
        #     fid.write("####### Model options #######\n")
        #     for data_type in opts:
        #         fid.write("%s: %s\n" % (data_type, str(opts[data_type])))

        #     fid.write("\n####### Network config #######\n")
        #     # fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
        #     # fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))

        #     fid.write("\n####### Training config #######\n")
        #     fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
        #     fid.write("%s: %s\n" % ('epochs', str(epochs)))
        #     fid.write("%s: %s\n" % ('lr', str(lr)))

        print('Wrote configs to {}'.format(config_path))

    def class_weights(self, apply_weights, sample_count):
        """
        Computes class weights for imbalanced data used during training
        Args:
            apply_weights: Whether to apply weights
            sample_count: Positive and negative sample counts
        Returns:
            A dictionary of class weights or None if no weights to be calculated
        """
        if not apply_weights:
            return None

        total = sample_count['neg_count'] + sample_count['pos_count']
        # formula from sklearn
        #neg_weight = (1 / sample_count['neg_count']) * (total) / 2.0
        #pos_weight = (1 / sample_count['pos_count']) * (total) / 2.0
        
        # use simple ratio
        neg_weight = sample_count['pos_count']/total
        pos_weight = sample_count['neg_count']/total

        print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        return {0: neg_weight, 1: pos_weight}

    def get_callbacks(self, learning_scheduler, model_path):
        """
        Creates a list of callabcks for training
        Args:
            learning_scheduler: Whether to use callbacks
        Returns:
            A list of call backs or None if learning_scheduler is false
        """
        callbacks = None

        # Set up learning schedulers
        if learning_scheduler:
            callbacks = []
            if 'early_stop' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'min_delta': 1.0, 'patience': 5,
                                  'verbose': 1}
                default_params.update(learning_scheduler['early_stop'])
                callbacks.append(EarlyStopping(**default_params))

            if 'plateau' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'factor': 0.2, 'patience': 5,
                                  'min_lr': 1e-08, 'verbose': 1}
                default_params.update(learning_scheduler['plateau'])
                callbacks.append(ReduceLROnPlateau(**default_params))

            if 'checkpoint' in learning_scheduler:
                default_params = {'filepath': model_path, 'monitor': 'val_loss',
                                  'save_best_only': True, 'save_weights_only': False,
                                  'save_freq': 'epoch', 'verbose': 2}
                default_params.update(learning_scheduler['checkpoint'])
                callbacks.append(ModelCheckpoint(**default_params))

        return callbacks

    def get_optimizer(self, optimizer):
        """
        Return an optimizer object
        Args:
            optimizer: The type of optimizer. Supports 'adam', 'sgd', 'rmsprop'
        Returns:
            An optimizer object
        """
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
        "{} optimizer is not implemented".format(optimizer)
        if optimizer.lower() == 'adam':
            return Adam
        elif optimizer.lower() == 'sgd':
            return SGD
        elif optimizer.lower() == 'rmsprop':
            return RMSprop

    def train(self, data_train,
              data_val=None,
              data_test=None,
              batch_size=32,
              epochs=60,
              lr=0.000005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')
        
        print(model_path)
        # print(type(model_path))
        # model_path = '/data1/PIE_2/' + model_path[12:]  ##change

        # print(model_path)

        # input('...')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size}) 
        # data_test = self.get_data('test', data_test, {**model_opts, 'batch_size': batch_size}) 
        # data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size}) 
        # asdasd

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]
                
        data_test = self.get_data('test', data_test, {**model_opts, 'batch_size': batch_size})['data']
        if self._generator:
            data_test = data_test[0]

        # print(data_test[0])
        # print(data_val[0])

        # input('wait......')

        # Create model
        train_model = self.get_model(data_train['data_params'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)


        train_model.compile(loss = BinaryFocalLoss(gamma = 5, pos_weight = 1), optimizer=optimizer, metrics=['accuracy'])
        # train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        train_model.summary()
        # input('wait...')
        
        
        ###CHANGE
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_weights_only=False, save_best_only=True, mode='max')
        callbacks = [checkpoint]
        ###########################

        # print(type(data_test))
        # print(data_test.keys())z

        # input('wait...')

        ###ORIG
        # callbacks = self.get_callbacks(learning_scheduler, model_path)
        # for dd in data_train['data'][0]:
        #     print('this is dd:', dd)
        #     print('type of dd:', type(dd))

        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_test,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        ###orig on
        # if 'checkpoint' not in learning_scheduler:
        #     print('Train model is saved to {}'.format(model_path))
        #     train_model.save(model_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        # model_opts_path = '/data1/PIE_2/' + model_opts_path[12:]  ##change after rebuild server __24.5.14
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        # config_path = '/data1/PIE_2/' + config_path[12:]  ##change ##change after rebuild server __24.5.14
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        # history_path = '/data1/PIE_2/' + history_path[12:]  ##change ##change after rebuild server __24.5.14
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        
        
        # saved_files_path = '/data1/PIE_2/' + saved_files_path[12:]  ##change ##change after rebuild server __24.5.14
        return saved_files_path

    # Test Functions
    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        
        # model_path_conf = 'data/models/' + model_path[13:]   ##change
        # print(model_path_conf)
        model_path_conf = model_path   ##change
        print(model_path_conf)


        with open(os.path.join(model_path_conf, 'configs.yaml'), 'r') as fid:    ##change
            opts = yaml.safe_load(fid)
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

        ###orig
        #test_model = load_model(os.path.join(model_path, 'model.h5'))
        test_model = load_model(os.path.join(model_path, 'model.h5'), custom_objects={'Transformer_Encoder': Transformer_Encoder, 'channel_attention': channel_attention})
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})
        

        # t1 = time.time()
        # test_results = test_model.predict(test_data['data'][0],
        #                                   batch_size=1, verbose=1)
        # t2 = time.time()
        # print(len(test_results))
        # print('Prediction time: {:.2f}'.format(t2 - t1))
        # print("Av FPS =", len(test_results)/(t2 - t1))
        print(len(test_data['data'][0]))
        wtf = test_data['data'][:][0]
        print(wtf[0].shape)
        input('wait...')
        for i in range(20):
            t1 = time.time()
            test_results = test_model.predict(test_data['data'][0][i],
                                            batch_size=1, verbose=1)
            t2 = time.time()
            print(len(test_results))
            print('Prediction time: {:.2f}'.format(t2 - t1))
            print("Av FPS =", len(test_results)/(t2 - t1))
        
        

        acc = accuracy_score(test_data['data'][1], np.round(test_results))
        f1 = f1_score(test_data['data'][1], np.round(test_results))
        auc = roc_auc_score(test_data['data'][1], np.round(test_results))
        roc = roc_curve(test_data['data'][1], test_results)
        precision = precision_score(test_data['data'][1], np.round(test_results))
        recall = recall_score(test_data['data'][1], np.round(test_results))
        pre_recall = precision_recall_curve(test_data['data'][1], test_results)
        
        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt':test_data['data'][1],
                         'y': test_results,
                         'image': test_data['image']}, picklefile)

        print('\n\n\n\n********>>>>>>>>>>>>> Result <<<<<<<<<<<<<<<<********')
        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))
        print('*********>>>>>>>>>>>>> Result End <<<<<<<<<<<<<<<<********\n\n\n\n')

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)

        Plist = []
        for p in data_test['pid']:
            Plist.append(p[0])

        # print(len(Plist))
        # print(len(test_data['data'][1]))
        # print(len(np.round(test_results)))
        
        Err = []
        for perID, gt, res in zip(Plist, test_data['data'][1], np.round(test_results)):
            if gt != res:
                Err.append([perID, gt, res])

        # print(Err)
        # print(len(Err))
        with open('wromgPid/' + model_path[-20:-1] + '_wromgPid.pkl','wb') as asd:   ###to dump the wrong id list
            pickle.dump(Err, asd)
            print('wrongPid dump to: ' + 'wromgPid/' + model_path[-20:-1] + '_wromgPid.pkl')

        # print(test_results)


        return acc, auc, f1, precision, recall

    def get_model(self, data_params):
        """
        Generates a model
        Args:
            data_params: Data parameters to use for model generation
        Returns:
            A model
        """
        raise NotImplementedError("get_model should be implemented")

    # Auxiliary function
    def _gru(self, name='gru', r_state=False, r_sequence=False):
        """
        A helper function to create a single GRU unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the GRU
            r_sequence: Whether to return a sequence
        Return:
            A GRU unit
        """
        return GRU(units=self._num_hidden_units,
                   return_state=r_state,
                   return_sequences=r_sequence,
                   stateful=False,
                   kernel_regularizer=self._regularizer,
                   recurrent_regularizer=self._regularizer,
                   bias_regularizer=self._regularizer,
                   name=name)

    def _lstm(self, name='lstm', r_state=False, r_sequence=False):
        """
        A helper function to create a single LSTM unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the LSTM
            r_sequence: Whether to return a sequence
        Return:
            A LSTM unit
        """
        return LSTM(units=self._num_hidden_units,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    kernel_regularizer=self._regularizer,
                    recurrent_regularizer=self._regularizer,
                    bias_regularizer=self._regularizer,
                    name=name)

    def create_stack_rnn(self, size, r_state=False, r_sequence=False):
        """
        Creates a stack of recurrent cells
        Args:
            size: The size of stack
            r_state: Whether to return the states of the GRU
            r_sequence: Whether the last stack layer to return a sequence
        Returns:
            A stacked recurrent model
        """
        cells = []
        for i in range(size):
            cells.append(self._rnn_cell(units=self._num_hidden_units,
                                        kernel_regularizer=self._regularizer,
                                        recurrent_regularizer=self._regularizer,
                                        bias_regularizer=self._regularizer, ))
        return RNN(cells, return_sequences=r_sequence, return_state=r_state)
    
    ####for yolo to add Head!!!
    #####HeadPose
    def process_detection( self, model, img, bbox):        
        y_min, x_min, y_max, x_max = bbox
        # # enlarge the bbox to include more background margin
        # y_min = max(0, y_min - abs(y_min - y_max) / 10)
        # y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
        # x_min = max(0, x_min - abs(x_min - x_max) / 5)
        # x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
        # x_max = min(x_max, img.shape[1])

        #img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print("this is img in process Detect:")
        # print(img)
        img_rgb = cv2.resize(img, (224, 224))
        img_rgb = np.expand_dims(img_rgb, axis=0)

        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
        yaw, pitch, roll = model.get_angle(img_rgb)
        yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
        draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )

        # if args.display == 'full':
        #     cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        #     cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        #     cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        return img, pitch, roll, yaw
        

    def addtwodimdict(self, thedict, key_a, key_b, val):
        if key_a in thedict.keys():
            thedict[key_a].update({key_b: val})
        else:
            thedict.update({key_a:{key_b: val}})
        
        return thedict

class SingleRNN(ActionPredict):
    """ A simple recurrent network """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn_type = cell_type

    def get_model(self, data_params):
        network_inputs = []
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        core_size = len(data_sizes)
        _rnn = self._gru if self._rnn_type == 'gru' else self._lstm

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i],
                                        name='input_' + data_types[i]))

        if len(network_inputs) > 1:
            inputs = Concatenate(axis=2)(network_inputs)
        else:
            inputs = network_inputs[0]

        encoder_output = _rnn(name='encoder')(inputs)

        encoder_output = Dense(1, activation='sigmoid',
                               name='output_dense')(encoder_output)
        net_model = Model(inputs=network_inputs,
                          outputs=encoder_output)

        return net_model


class StackedRNN(ActionPredict):
    """ A stacked recurrent prediction model based on
    Yue-Hei et al. "Beyond short snippets: Deep networks for video classification."
    CVPR, 2015." """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params):
        network_inputs = []
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        core_size = len(data_sizes)
        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

        if len(network_inputs) > 1:
            inputs = Concatenate(axis=2)(network_inputs)
        else:
            inputs = network_inputs[0]

        encoder_output = self.create_stack_rnn(core_size)(inputs)
        encoder_output = Dense(1, activation='sigmoid',
                               name='output_dense')(encoder_output)
        net_model = Model(inputs=network_inputs,
                          outputs=encoder_output)
        return net_model


class MultiRNN(ActionPredict):
    """
    A multi-stream recurrent prediction model inspired by
    Bhattacharyya et al. "Long-term on-board prediction of people in traffic
    scenes under uncertainty." CVPR, 2018.
    """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(self._rnn(name='enc_' + data_types[i])(network_inputs[i]))

        if len(encoder_outputs) > 1:
            encodings = Concatenate(axis=1)(encoder_outputs)
        else:
            encodings = encoder_outputs[0]

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense')(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        return net_model


class HierarchicalRNN(ActionPredict):
    """
    A Hierarchical recurrent prediction model inspired by
    Du et al. "Hierarchical recurrent neural network for skeleton
    based action recognition." CVPR, 2015.
    """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(
                self._rnn(name='enc_' + data_types[i], r_sequence=True)(network_inputs[i]))

        if len(network_inputs) > 1:
            inputs = Concatenate(axis=2)(encoder_outputs)
        else:
            inputs = network_inputs[0]

        second_layer = self._rnn(name='final_enc', r_sequence=False)(inputs)

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense')(second_layer)
        net_model = Model(inputs=network_inputs,
                          outputs=model_output)

        return net_model


class SFRNN(ActionPredict):
    """
    Pedestrian crossing prediction based on
    Rasouli et al. "Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs."
    BMVC, 2020. The original code can be found at https://github.com/aras62/SF-GRU
    """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        return_sequence = True
        num_layers = len(data_sizes)

        for i in range(num_layers):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

            if i == num_layers - 1:
                return_sequence = False

            if i == 0:
                x = self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(network_inputs[i])
            else:
                x = Concatenate(axis=2)([x, network_inputs[i]])
                x = self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(x)

        model_output = Dense(1, activation='sigmoid', name='output_dense')(x)

        net_model = Model(inputs=network_inputs, outputs=model_output)

        return net_model


class C3D(ActionPredict):
    """
    C3D code based on
    Tran et al. "Learning spatiotemporal features with 3d convolutional networks.",
    CVPR, 2015. The code is based on implementation availble at
    https://github.com/adamcasson/c3d
    """
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (112, 112)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super(C3D, self).get_data(data_type, data_raw, model_opts)

    # TODO: use keras function to load weights
    def get_model(self, data_params):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        if not os.path.exists(self._weights):
            weights_url = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'
            wget.download(weights_url, self._weights)
        net_model = C3DNet(freeze_conv_layers=self._freeze_conv_layers,
                           dropout=self._dropout,
                           dense_activation=self._dense_activation,
                           include_top=True,
                           weights=self._weights)
        net_model.summary()

        return net_model


class I3D(ActionPredict):
    """
    A single I3D method based on
    Carreira et al. "Quo vadis, action recognition? a new model and the kinetics dataset."
    CVPR 2017. This model is based on the original code published by the authors which
    can be found at https://github.com/deepmind/kinetics-i3d
    """
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'i3d'

    def get_data(self, data_type, data_raw, model_opts):
        assert len(model_opts['obs_input_type']) == 1
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        model_opts['process'] = False
        model_opts['backbone'] = 'i3d'
        return super(I3D, self).get_data(data_type, data_raw, model_opts)

    # TODO: use keras function to load weights
    def get_model(self, data_params):
        # TODO: use keras function to load weights

        if 'flow' in data_params['data_types'][0]:
            weights_url = 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
            self._weights='weights/i3d_flow_weights.h5'
            num_channels = 2
        else:
            weights_url = 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
            num_channels = 3
            self._weights='weights/i3d_rgb_weights.h5'
            
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        if not os.path.exists(self._weights):
            wget.download(weights_url, self._weights)
        
        net_model = I3DNet(freeze_conv_layers=self._freeze_conv_layers, weights=self._weights,
                           dense_activation=self._dense_activation, dropout=self._dropout,
                           num_channels=num_channels, include_top=True)

        net_model.summary()
        return net_model


class TwoStreamI3D(ActionPredict):
    """
    Two-stream 3D method based on
    Carreira et al. "Quo vadis, action recognition? a new model and the kinetics dataset."
    CVPR 2017. This model is based on the original code published by the authors which
    can be found at https://github.com/deepmind/kinetics-i3d
    """
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights_rgb='weights/i3d_rgb_weights.h5',
                 weights_flow='weights/i3d_flow_weights.h5',
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights_rgb: Pre-trained weights for rgb stream.
            weights_flow: Pre-trained weights for optical flow stream.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights_rgb = weights_rgb
        self._weights_flow = weights_flow
        self._weights = None
        self._backbone = 'i3d'

    def get_data(self, data_type, data_raw, model_opts):
        assert len(model_opts['obs_input_type']) == 1
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        model_opts['process'] = False
        model_opts['backbone'] = 'i3d'
        return super(TwoStreamI3D, self).get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params):
        # TODO: use keras function to load weights
        if 'flow' in data_params['data_types'][0]:
            weights_url = 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
            num_channels = 2
        else:
            weights_url = 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
            num_channels = 3
        
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        if not os.path.exists(self._weights):
            wget.download(weights_url, self._weights)

        net_model = I3DNet(freeze_conv_layers=self._freeze_conv_layers, weights=self._weights,
                           dense_activation=self._dense_activation, dropout=self._dropout,
                           num_channels=num_channels, include_top=True)

        net_model.summary()
        return net_model

    def train(self, data_train,
              data_val=None,
              batch_size=4,
              epochs=60,
              lr=0.000005,
              optimizer='sgd',
              learning_scheduler=None,
              model_opts=None):

        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}

        model_opts['reshape'] = True
        local_params = {k: v for k, v in locals().items() if k != 'self'}

        #####  Optical flow model
        self.train_model('opt_flow', **local_params)

        ##### rgb model
        self.train_model('rgb', **local_params)

        # Save settings
        model_opts_path, saved_files_path = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)
        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        return saved_files_path

    def train_model(self, model_type, path_params, learning_scheduler, data_train, data_val,
                    optimizer, batch_size, model_opts,  epochs, lr, **kwargs):
        """
        Trains a single model
        Args:
            model_type: The type of model, 'rgb' or 'opt_flow'
            path_params: Parameters for generating paths for saving models and configurations
            callbacks: List of training call back functions
            model_type: The model type, 'rgb' or 'opt_flow'
            For other parameters refer to train()
        """
        learning_scheduler = learning_scheduler or {}
        self._weights = self._weights_rgb if model_type == 'rgb' else self._weights_flow

        _opts = model_opts.copy()
        if model_type == 'opt_flow':
            _opts['obs_input_type'] = [_opts['obs_input_type'][0] + '_flow']

        # Read train data
        data_train = self.get_data('train', data_train, {**_opts, 'batch_size': batch_size})
        if data_val is not None:
            data_val = self.get_data('val', data_val, {**_opts, 'batch_size': batch_size})
            data_val = data_val['data']
            if self._generator:
                data_val = data_val[0]

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model = self.get_model(data_train['data_params'])
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        data_path_params = {**path_params, 'sub_folder': model_type}
        model_path, _ = get_path(**data_path_params, file_name='model.h5')
        callbacks = self.get_callbacks(learning_scheduler,model_path)

        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('{} train model is saved to {}'.format(model_type, model_path))
            train_model.save(model_path)
        # Save training history
        history_path, saved_files_path = get_path(**data_path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)


    def test(self, data_test, model_path=''):
        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        # Evaluate rgb model
        test_data_rgb = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})
        rgb_model = load_model(os.path.join(model_path, 'rgb', 'model.h5'))
        results_rgb = rgb_model.predict(test_data_rgb['data'][0], verbose=1)

        model_opts['obs_input_type'] = [model_opts['obs_input_type'][0] + '_flow']
        test_data_flow = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})
        opt_flow_model = load_model(os.path.join(model_path, 'opt_flow', 'model.h5'))
        results_opt_flow = opt_flow_model.predict(test_data_flow['data'][0], verbose=1)

        # Average the predictions for both streams
        results = (results_rgb + results_opt_flow) / 2.0
        gt = test_data_rgb['data'][1]

        acc = accuracy_score(gt, np.round(results))
        f1 = f1_score(gt, np.round(results))
        auc = roc_auc_score(gt, np.round(results))
        roc = roc_curve(gt, results)
        precision = precision_score(gt, np.round(results))
        recall = recall_score(gt, np.round(results))
        pre_recall = precision_recall_curve(gt, results)

        print('acc:{:.2f} auc:{:0.2f} f1:{:0.2f} precision:{:0.2f} recall:{:0.2f}'.format(acc, auc, f1, precision,
                                                                                          recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

        # with open(save_results_path, 'w') as fid:
        #     yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


class Static(ActionPredict):
    """
    A static model which uses features from the last convolution
    layer and a dense layer to classify
    """
    def __init__(self,
                 dropout=0.0,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='imagenet',
                 num_classes=1,
                 backbone='vgg16',
                 global_pooling='avg',
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
            num_classes: Number of activity classes to predict.
            backbone: Backbone network. Only vgg16 is supported.
            global_pooling: Global pooling method used for generating convolutional features
        """
        super().__init__(**kwargs)
        # Network parameters

        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._num_classes = num_classes
        self._pooling = global_pooling
        self._conv_models = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50, 'alexnet': AlexNet}
        self._backbone = backbone

    def get_data(self, data_type, data_raw, model_opts):
        """
        Generates train/test data
        :param data_raw: The sequences received from the dataset interface
        :param model_opts: Model options:
                            'obs_input_type': The types of features to be used for train/test. The order
                                            in which features are named in the list defines at what level
                                            in the network the features are processed. e.g. ['local_context',
                                            pose] would behave different to ['pose', 'local_context']
                            'enlarge_ratio': The ratio (with respect to bounding boxes) that is used for processing
                                           context surrounding pedestrians.
                            'pred_target_type': Learning target objective. Currently only supports 'crossing'
                            'obs_length': Observation length prior to reasoning
                            'time_to_event': Number of frames until the event occurs
                            'dataset': Name of the dataset

        :return: Train/Test data
        """
        # Stack of 5-10 optical flow. For each  image average results over two
        # branches and average for all samples from the sequence
        # single images and stacks of optical flow

        # assert len(model_opts['obs_input_type']) == 1
        self._generator = model_opts.get('generator', True)
        model_opts['normalize_boxes'] = False
        process = False
        aux_name = '_'.join([self._backbone, 'raw']).strip('_')
        dataset = model_opts['dataset']
        eratio = model_opts['enlarge_ratio']

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        feature_type = model_opts['obs_input_type'][0]

        assert feature_type in ['local_box', 'local_context', 'scene']

        _data_samples = {}
        _data_samples['crossing'] = data['crossing']
        data_type_sizes_dict = {}


        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')

        save_folder_name = '_'.join(['local_context', aux_name, str(eratio)]) if feature_type == 'local_context' \
                           else '_'.join(['local_box', aux_name])
        path_to_features, _ = get_path(save_folder=save_folder_name,
                                          dataset=dataset,
                                          save_root_folder='data/features')
        data_gen_params = {'data_type': data_type, 'save_path': path_to_features,
                           'crop_type': 'none', 'process': process}

        if feature_type == 'local_box':
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif feature_type == 'local_context':
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio

        for k, v in data.items():
            if 'act' not in k:
                if len(v.shape) == 3:
                    data[k] = np.expand_dims(v[:, -1, :], axis=1)
                else:
                    data[k] = np.expand_dims(v[:, -1], axis=-1)


        _data_samples[feature_type], feat_shape = self.load_images_crop_and_process(data['image'],
                                                                      data['box_org'],
                                                                      data['ped_id'],
                                                                      **data_gen_params)
        if not self._generator:
            _data_samples[feature_type] = np.squeeze(_data_samples[feature_type])
        data_type_sizes_dict[feature_type] = feat_shape[1:]

        # create the final data file to be returned
        if self._generator:
            _data_rgb = (DataGenerator(data=[_data_samples[feature_type]],
                                   labels=data['crossing'],
                                   data_sizes=[data_type_sizes_dict[feature_type]],
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), _data_samples['crossing'])  # set y to None

        else:
            _data_rgb = (_data_samples[feature_type], _data_samples['crossing'])

        return {'data': _data_rgb,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': [feature_type],
                                'data_sizes': [data_type_sizes_dict[feature_type]]},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        data_size = data_params['data_sizes'][0]
        context_net = self._conv_models[self._backbone](input_shape=data_size,
                                                        include_top=False, weights=self._weights,
                                                        pooling=self._pooling)
        output = Dense(self._num_classes,
                       activation=self._dense_activation,
                       name='output_dense')(context_net.outputs[0])
        net_model = Model(inputs=context_net.inputs[0], outputs=output)

        net_model.summary()
        return net_model


class ConvLSTM(ActionPredict):
    """
    A Convolutional LSTM model for sequence learning
    """
    def __init__(self,
                 global_pooling='avg',
                 filter=64,
                 kernel_size=1,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Global pooling method used for generating convolutional features
            filter: Number of conv filters
            kernel_size: Kernel size of conv filters
            dropout: Dropout value for fc6-7 only for alexnet.
            recurrent_dropout: Recurrent dropout value
        """
        super().__init__(**kwargs)
        # Network parameters
        self._pooling = global_pooling
        self._filter = filter
        self._kernel_size = kernel_size
        self._dropout = dropout
        self._recurrent_dropout = recurrent_dropout
        self._backbone = ''

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        model_opts['normalize_boxes'] = False
        model_opts['process'] = False
        model_opts['backbone'] = ''
        return super(ConvLSTM, self).get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params):
        data_size = data_params['data_sizes'][0]
        data_type = data_params['data_types'][0]

        x_in = Input(shape=data_size, name='input_' + data_type)
        convlstm = ConvLSTM2D(filters=self._filter, kernel_size=self._kernel_size,
                              kernel_regularizer=self._regularizer, recurrent_regularizer=self._regularizer,
                              bias_regularizer=self._regularizer, dropout=self._dropout,
                              recurrent_dropout=self._recurrent_dropout)(x_in)
        if self._pooling == 'avg':
            out = GlobalAveragePooling2D()(convlstm)
        elif self._pooling == 'max':
            out = GlobalMaxPooling2D()(convlstm)
        else:
            out = Flatten(name='flatten')(convlstm)

        _output = Dense(1, activation='sigmoid', name='output_dense')(out)
        net_model = Model(inputs=x_in, outputs=_output)
        net_model.summary()
        return net_model


class ATGC(ActionPredict):
    """
    This is an implementation of pedestrian crossing prediction model based on
    Rasouli et al. "Are they going to cross? A benchmark dataset and baseline
    for pedestrian crosswalk behavior.", ICCVW, 2017.
    """
    def __init__(self,
                 dropout=0.0,
                 freeze_conv_layers=False,
                 weights='imagenet',
                 backbone='alexnet',
                 global_pooling='avg',
                 **kwargs):
        """
            Class init function
            Args:
                dropout: Dropout value for fc6-7 only for alexnet.
                freeze_conv_layers: If set true, only fc layers of the networks are trained
                weights: Pre-trained weights for networks.
                backbone: Backbone network. Only vgg16 is supported.
                global_pooling: Global pooling method used for generating convolutional features
        """
        super().__init__(**kwargs)
        self._dropout = dropout
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._pooling = global_pooling
        self._conv_models = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50, 'alexnet': AlexNet}
        self._backbone = backbone

    def get_scene_tags(self, traffic_labels):
        """
        Generates a 1-hot vector for traffic labels
        Args:
            traffic_labels: A dictionary of traffic labels read from the label
            Original labels are:
            'ped_crossing','ped_sign','traffic_light','stop_sign','road_type','num_lanes'
             traffic_light: 0: n/a, 1: red, 2: green
             street: 0, parking_lot: 1, garage: 2
             final_output: [narrow_road, wide_road, ped_sign, ped_crossing,
                            stop_sign, traffic_light, parking_lot]
        Returns:
            List of 1-hot vectors for traffic labels
        """

        scene_tags = []
        for seq in traffic_labels:
            step_tags = []
            for step in seq:
                tags = [int(step[0]['num_lanes'] <= 2), int(step[0]['num_lanes'] > 2),
                        step[0]['ped_sign'], step[0]['ped_crossing'], step[0]['stop_sign'],
                        int(step[0]['traffic_light'] > 0), int(step[0]['road_type'] == 1)]
                step_tags.append(tags)
            scene_tags.append(step_tags)
        return scene_tags

    def get_data_sequence(self, data_type, data_raw, opts):
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')

        d = {'box': data_raw['bbox'].copy(),
             'crossing': data_raw['activities'].copy(),
             'walking': data_raw['actions'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'looking': data_raw['looks'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        d['scene'] = self.get_scene_tags(data_raw['traffic'])
        d['tte'] = []
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])

        else:
            overlap = opts['overlap'] if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res

            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx, olap_res)])
                d[k] = seqs
            tte_seq = []
            for seq in d['box']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                tte_seq.extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
            d['tte'] = tte_seq
        for k in d.keys():
            d[k] = np.array(d[k])

        if opts['pred_target_type'] != 'scene':
            if opts['pred_target_type'] == 'crossing':
                dcount = d[opts['pred_target_type']][:, 0, :]
            else:
                dcount = d[opts['pred_target_type']].reshape((-1, 1))
            pos_count = np.count_nonzero(dcount)
            neg_count = len(dcount) - pos_count
            print("{} : Negative {} and positive {} sample counts".format(opts['pred_target_type'],
                                                                          neg_count, pos_count))
        else:
            pos_count, neg_count = 0, 0
        return d, neg_count, pos_count

    def get_data(self, data_type, data_raw, model_opts):
        model_opts['normalize_boxes'] = False
        process = False
        aux_name = '_'.join([self._backbone, 'raw']).strip('_')
        dataset = model_opts['dataset']
        self._generator = model_opts.get('generator', False)

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        _data_samples = {}
        data_type_sizes_dict = {}

        # crop only bounding boxes
        if 'ped_legs' in model_opts['obs_input_type']:
            print('\n#####################################')
            print('Generating pedestrian leg samples {}'.format(data_type))
            print('#####################################')
            path_to_ped_legs, _ = get_path(save_folder='_'.join(['ped_legs', aux_name]),
                                           dataset=dataset,
                                           save_root_folder='data/features')
            leg_coords = np.copy(data['box_org'])
            for seq in leg_coords:
                for bbox in seq:
                    height = bbox[3] - bbox[1]
                    bbox[1] = bbox[1] + height // 2
            target_dim = (227, 227) if self._backbone == 'alexnet' else (224,224)
            data['ped_legs'], feat_shape = self.load_images_crop_and_process(data['image'],
                                                                 leg_coords,
                                                                 data['ped_id'],
                                                                 data_type=data_type,
                                                                 save_path=path_to_ped_legs,
                                                                 crop_type='bbox',
                                                                 crop_mode='warp',
                                                                 target_dim=target_dim,
                                                                 process=process)

            data_type_sizes_dict['ped_legs'] = feat_shape
        if 'ped_head' in model_opts['obs_input_type']:
            print('\n#####################################')
            print('Generating pedestrian head samples {}'.format(data_type))
            print('#####################################')

            path_to_ped_heads, _ = get_path(save_folder='_'.join(['ped_head', aux_name]),
                                              dataset=dataset,
                                              save_root_folder='data/features')
            head_coords = np.copy(data['box_org'])
            for seq in head_coords:
                for bbox in seq:
                    height = bbox[3] - bbox[1]
                    bbox[3] = bbox[3] - (height * 2) // 3
            target_dim = (227, 227) if self._backbone == 'alexnet' else (224,224)
            data['ped_head'], feat_shape = self.load_images_crop_and_process(data['image'],
                                                                 head_coords,
                                                                 data['ped_id'],
                                                                 data_type=data_type,
                                                                 save_path=path_to_ped_heads,
                                                                 crop_type='bbox',
                                                                 crop_mode='warp',
                                                                 target_dim=target_dim,
                                                                 process=process)
            data_type_sizes_dict['ped_head'] = feat_shape
        if 'scene_context' in model_opts['obs_input_type']:
            print('\n#####################################')
            print('Generating local context {}'.format(data_type))
            print('#####################################')
            target_dim = (540, 540) if self._backbone == 'alexnet' else (224, 224)
            path_to_scene_context, _ = get_path(save_folder='_'.join(['scene_context', aux_name]),
                                                dataset=dataset,
                                                save_root_folder='data/features')
            data['scene_context'], feat_shape = self.load_images_crop_and_process(data['image'],
                                                                      data['box_org'],
                                                                      data['ped_id'],
                                                                      data_type=data_type,
                                                                      save_path=path_to_scene_context,
                                                                      crop_type='none',
                                                                      target_dim=target_dim,
                                                                      process=process)
            data_type_sizes_dict['scene_context'] = feat_shape

        # Reshape the sample tracks by collapsing sequence size to the number of samples
        # (samples, seq_size, features) -> (samples*seq_size, features)
        if model_opts.get('reshape', False):
            for k in data:
                dsize = data_type_sizes_dict.get(k, data[k].shape)
                if self._generator:
                    new_shape = (-1, data[k].shape[-1]) if data[k].ndim > 2 else (-1, 1)
                else:
                    new_shape = (-1,) + dsize[1:] if len(dsize) > 3  else (-1, dsize[-1])
                data[k] = np.reshape(data[k], new_shape)
                data_type_sizes_dict[k] = dsize[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            _data.append(data[d_type])
            data_sizes.append(data_type_sizes_dict[d_type])
            data_types.append(d_type)

        classes = 7 if model_opts['pred_target_type'] == 'scene' else 2

        # create the final data file to be returned
        if self._generator:
            is_train_data = True
            if data_type == 'test' or model_opts.get('predict_data', False):
                is_train_data = False
            data_inputs = []
            for i, d in enumerate(_data):
                data_inputs.append(DataGenerator(data=[d],
                                   labels=data[model_opts['pred_target_type']],
                                   data_sizes=[data_sizes[i]],
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=[model_opts['obs_input_type'][i]],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=is_train_data,
                                   to_fit=is_train_data))
            _data = (data_inputs, data[model_opts['pred_target_type']]) # set y to None
        else:
            _data = (_data, data[model_opts['pred_target_type']])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes,
                                'pred_type': model_opts['pred_target_type'],
                                'num_classes': classes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=20,
              learning_scheduler=None,
              model_opts=None,
              **kwargs):
        model_opts['model_folder_name'] = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        model_opts['reshape'] = True

        model_opts['pred_target_type'] = 'walking'
        model_opts['obs_input_type'] = ['ped_legs']
        walk_model = self.train_model(data_train, data_val,
                                      learning_scheduler=learning_scheduler,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      model_opts=model_opts)

        model_opts['pred_target_type'] = 'looking'
        model_opts['obs_input_type'] = ['ped_head']
        look_model = self.train_model(data_train, data_val,
                                      learning_scheduler=learning_scheduler,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      model_opts=model_opts)



        model_opts['obs_input_type'] = ['scene_context']
        model_opts['pred_target_type'] = 'scene'
        scene_model = self.train_model(data_train, data_val,
                                       learning_scheduler=learning_scheduler,
                                       epochs=epochs, lr=0.000625,
                                       batch_size=batch_size,
                                       loss_func='categorical_crossentropy',
                                       activation='sigmoid',
                                       model_opts=model_opts)

        model_opts['model_paths'] = {'looking': look_model, 'walking': walk_model, 'scene': scene_model}
        saved_files_path = self.train_final(data_train, model_opts=model_opts)

        return saved_files_path

    def train_model(self, data_train,
                    data_val=None, batch_size=32,
                    epochs=60, lr=0.00001,
                    optimizer='sgd',
                    loss_func='sparse_categorical_crossentropy',
                    activation='sigmoid',
                    learning_scheduler =None,
                    model_opts=None):
        """
        Trains a single model
        Args:
            data_train: Training data
            data_val: Validation data
            loss_func: The type of loss function to use
            activation: The activation type for the last (predictions) layer
            For other parameters refer to train()
        """
        learning_scheduler = learning_scheduler or {}

        # Set the path for saving models
        model_folder_name = model_opts.get('model_folder_name',
                                           time.strftime("%d%b%Y-%Hh%Mm%Ss"))
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset'],
                       'sub_folder': model_opts['pred_target_type']}
        model_path, _ = get_path(**path_params, file_name='model.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})
        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0][0]

        # Create model
        data_train['data_params']['activation'] = activation
        train_model = self.get_model(data_train['data_params'])

        # Train the model
        if data_train['data_params']['num_classes'] > 2:
            model_opts['apply_class_weights'] = False
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        callbacks = self.get_callbacks(learning_scheduler, model_path)

        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
        history = train_model.fit(x=data_train['data'][0][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  callbacks=callbacks,
                                  verbose=2)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def train_final(self, data_train,
                    batch_size=1,
                    num_iterations=1000,
                    model_opts=None):
        """
        Trains the final SVM model
        Args:
            data_train: Training data
            batch_size: Batch sizes used for generator
            num_iterations: Number of iterations for SVM model
            model_opts: Model options
        Returns:
            The path to the root folder where models are saved
        """
        print("Training final model!")
        # Set the path for saving models
        model_folder_name = model_opts.get('model_folder_name',
                                           time.strftime("%d%b%Y-%Hh%Mm%Ss"))
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}

        # Read train data
        model_opts['obs_input_type'] = ['ped_head', 'ped_legs', 'scene_context']
        model_opts['pred_target_type'] = 'crossing'

        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size,
                                                         'predict_data': True})

        if data_train['data_params']['num_classes'] > 2:
            model_opts['apply_class_weights'] = False
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])

        # Load conv model
        look_model = self.get_model({'data_sizes': [data_train['data_params']['data_sizes'][0]],
                                     'weights': model_opts['model_paths']['looking'], 'features': True})
        looking_features = look_model.predict(data_train['data'][0][0], verbose=1)

        walk_model = self.get_model({'data_sizes': [data_train['data_params']['data_sizes'][1]],
                                     'weights': model_opts['model_paths']['walking'], 'features': True})
        walking_features = walk_model.predict(data_train['data'][0][1], verbose=1)

        scene_model = self.get_model({'data_sizes': [data_train['data_params']['data_sizes'][2]],
                                     'weights': model_opts['model_paths']['scene'], 'features': True,
                                      'num_classes': 7})
        scene_features = scene_model.predict(data_train['data'][0][2],  verbose=1)

        svm_features = np.concatenate([looking_features, walking_features, scene_features], axis=-1)

        svm_model = make_pipeline(StandardScaler(),
                                  LinearSVC(random_state=0, tol=1e-5,
                                            max_iter=num_iterations,
                                            class_weight=class_w))
        svm_model.fit(svm_features, np.squeeze(data_train['data'][1]))

        # Save configs
        model_path, saved_files_path = get_path(**path_params, file_name='model.pkl')
        with open(model_path, 'wb') as fid:
            pickle.dump(svm_model, fid, pickle.HIGHEST_PROTOCOL)

        # Save configs
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path


    def get_model(self, data_params):
        K.clear_session()
        net_model = self._conv_models[self._backbone](input_shape=data_params['data_sizes'][0],
                            include_top=False, weights=self._weights)

        # Convert to fully connected
        net_model = convert_to_fcn(net_model, classes=data_params.get('num_classes', 2),
                       activation=data_params.get('activation', 'softmax'),
                       pooling=self._pooling, features=data_params.get('features', False))
        net_model.summary()
        return net_model


    def test(self, data_test, model_path=''):
        """
        Test function
        :param data_test: The raw data received from the dataset interface
        :param model_path: The path to the folder where the model and config files are saved.
        :return: The following performance metrics: acc, auc, f1, precision, recall
        """
        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        data_test = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})

        # Load conv model
        look_model = self.get_model({'data_sizes': [data_test['data_params']['data_sizes'][0]],
                                     'weights': model_opts['model_paths']['looking'], 'features': True})
        walk_model = self.get_model({'data_sizes': [data_test['data_params']['data_sizes'][1]],
                                     'weights': model_opts['model_paths']['walking'], 'features': True})
        scene_model = self.get_model({'data_sizes': [data_test['data_params']['data_sizes'][2]],
                                      'weights': model_opts['model_paths']['scene'], 'features': True,
                                      'num_classes': 7})

        with open(os.path.join(model_path, 'model.pkl'), 'rb') as fid:
            try:
                svm_model = pickle.load(fid)
            except:
                svm_model = pickle.load(fid, encoding='bytes')

        looking_features = look_model.predict(data_test['data'][0][0], verbose=1)
        walking_features = walk_model.predict(data_test['data'][0][1], verbose=1)
        scene_features = scene_model.predict(data_test['data'][0][2], verbose=1)
        svm_features = np.concatenate([looking_features, walking_features, scene_features], axis=-1)
        res = svm_model.predict(svm_features)
        res = np.reshape(res, (-1, model_opts['obs_length'], 1))
        results = np.mean(res, axis=1)

        gt = np.reshape(data_test['data'][1], (-1, model_opts['obs_length'], 1))[:, 1, :]
        acc = accuracy_score(gt, np.round(results))
        f1 = f1_score(gt, np.round(results))
        auc = roc_auc_score(gt, np.round(results))
        roc = roc_curve(gt, results)
        precision = precision_score(gt, np.round(results))
        recall = recall_score(gt, np.round(results))
        pre_recall = precision_recall_curve(gt, results)

        data_tte = np.squeeze(data_test['tte'][:len(gt)])

        print('acc:{:.2f} auc:{:0.2f} f1:{:0.2f} precision:{:0.2f} recall:{:0.2f}'.format(acc, auc, f1, precision,
                                                                                          recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

        with open(save_results_path, 'w') as fid:
            yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


class TwoStream(ActionPredict):
    """
    This is an implementation of two-stream network based on
    Simonyan et al. "Two-stream convolutional networks for action recognition
    in videos.", NeurIPS, 2014.
    """
    def __init__(self,
                 dropout=0.0,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='imagenet',
                 backbone='vgg16',
                 num_classes=1,
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
            num_classes: Number of activity classes to predict.
            backbone: Backbone network. Only vgg16 is supported.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._num_classes = num_classes
        if backbone != 'vgg16':
            print("Only vgg16 backbone is supported")
            backbone ='vgg16'
        self._backbone = backbone
        self._conv_model = vgg16.VGG16

    def get_data_sequence(self, data_type, data_raw, opts):

        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'box': data_raw['bbox'].copy(),
             'crossing': data_raw['activities'].copy(),
             'walking': data_raw['actions'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'looking': data_raw['looks'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
        else:
            overlap = opts['overlap'] if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res

            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs
            for seq in d['box']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
        for k in d.keys():
            d[k] = np.array(d[k])

        dcount = d['crossing'][:, 0, :]
        pos_count = np.count_nonzero(dcount)
        neg_count = len(dcount) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def get_data(self, data_type, data_raw, model_opts):
        model_opts['normalize_boxes'] = False
        aux_name = '_'.join([self._backbone, 'raw']).strip('_')
        process = False
        dataset = model_opts['dataset']
        eratio = model_opts['enlarge_ratio']
        self._generator = model_opts.get('generator', False)

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        feature_type = model_opts['obs_input_type'][0]

        # Only 3 types of rgb features are supported
        assert feature_type in ['local_box', 'local_context', 'scene']

        _data_samples = {'crossing': data['crossing']}
        data_type_sizes_dict = {}

        data_gen_params = {'data_type': data_type, 'crop_type': 'none'}

        if feature_type == 'local_box':
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif feature_type == 'local_context':
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio

        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')

        save_folder_name = '_'.join([feature_type, aux_name, str(eratio)]) \
                           if feature_type in ['local_context', 'local_surround'] \
                           else '_'.join([feature_type, aux_name])
        path_to_features, _ = get_path(save_folder=save_folder_name,
                                       dataset=dataset,
                                       save_root_folder='data/features')
        data_gen_params['save_path'] = path_to_features

        # Extract relevant frames based on the optical flow length
        ofl = model_opts.get('optical_flow_length', 10)
        stidx = ofl - round((ofl + 1) / 2)
        endidx = (ofl + 1) // 2

        # data_type_sizes_dict[feature_type] = (_data_samples[feature_type].shape[1], *feat_shape[1:])
        _data_samples['crossing'] = _data_samples['crossing'][:, stidx:-endidx, ...]
        effective_dimension = _data_samples['crossing'].shape[1]

        _data_samples[feature_type], feat_shape = self.load_images_crop_and_process(data['image'][:, stidx:-endidx, ...],
                                                                                    data['box_org'][:, stidx:-endidx, ...],
                                                                                    data['ped_id'][:, stidx:-endidx, ...],
                                                                                    process=process,
                                                                                    **data_gen_params)
        data_type_sizes_dict[feature_type] = feat_shape

        print('\n#####################################')
        print('Generating optical flow {} {}'.format(feature_type, data_type))
        print('#####################################')

        save_folder_name = '_'.join([feature_type, 'flow', str(eratio)]) \
                           if feature_type == 'local_context' else '_'.join([feature_type, 'flow'])
        path_to_features, _ = get_path(save_folder=save_folder_name,
                                       dataset=dataset,
                                       save_root_folder='data/features')
        data_gen_params['save_path'] = path_to_features
        _data_samples['optical_flow'], feat_shape = self.get_optical_flow(data['image'],
                                                                          data['box_org'],
                                                                          data['ped_id'],
                                                                          **data_gen_params)

        # Create opflow data by stacking batches of optflow
        opt_flow = []
        if self._generator:
            _data_samples['optical_flow'] = np.expand_dims(_data_samples['optical_flow'], axis=-1)

        for sample in _data_samples['optical_flow']:
            opf = [np.concatenate(sample[i:i+ofl, ...], axis=-1) for i in range(sample.shape[0] - ofl + 1)]
            opt_flow.append(opf)
        _data_samples['optical_flow'] = np.array(opt_flow)
        if self._generator:
            data_type_sizes_dict['optical_flow'] = (feat_shape[0] - ofl + 1,
                                                    *feat_shape[1:3], feat_shape[3]*ofl)
        else:
            data_type_sizes_dict['optical_flow'] = _data_samples['optical_flow'].shape[1:]

        if model_opts.get('reshape', False):
            for k in _data_samples:
                dsize = data_type_sizes_dict.get(k, _data_samples[k].shape)
                if self._generator:
                    new_shape = (-1, _data_samples[k].shape[-1]) if _data_samples[k].ndim > 2 else (-1, 1)
                else:
                    new_shape = (-1,) + dsize[1:] if len(dsize) > 3 else (-1, dsize[-1])
                _data_samples[k] = np.reshape(_data_samples[k], new_shape)
                data_type_sizes_dict[k] = dsize[1:]
        # create the final data file to be returned
        if self._generator:
            _data_rgb = (DataGenerator(data=[_data_samples[feature_type]],
                                   labels=_data_samples['crossing'],
                                   data_sizes=[data_type_sizes_dict[feature_type]],
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), _data_samples['crossing'])  # set y to None
            _data_opt_flow = (DataGenerator(data=[_data_samples['optical_flow']],
                                   labels=_data_samples['crossing'],
                                   data_sizes=[data_type_sizes_dict['optical_flow']],
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=['optical_flow'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test',
                                   stack_feats=True), _data_samples['crossing'])  # set y to None
        else:
            _data_rgb = (_data_samples[feature_type], _data_samples['crossing'])
            _data_opt_flow = (_data_samples['optical_flow'], _data_samples['crossing'])

        return {'data_rgb': _data_rgb,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'data_opt_flow': _data_opt_flow,
                'data_params_rgb': {'data_types': [feature_type],
                                    'data_sizes': [data_type_sizes_dict[feature_type]]},
                'data_params_opt_flow': {'data_types': ['optical_flow'],
                                         'data_sizes': [data_type_sizes_dict['optical_flow']]},
                'effective_dimension': effective_dimension,
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def add_dropout(self, model, add_new_pred=False):
        """
           Adds dropout layers to a given vgg16 network. If specified, changes the dimension of
           the last layer (predictions)
           Args:
               model: A given vgg16 model
               add_new_pred: Whether to change the final layer
           Returns:
               Returns the new model
        """
        # Change to a single class output and add dropout
        fc1_dropout = Dropout(self._dropout)(model.layers[-3].output)
        fc2 = model.layers[-2](fc1_dropout)
        fc2_dropout = Dropout(self._dropout)(fc2)
        if add_new_pred:
            output = Dense(self._num_classes, name='predictions', activation='sigmoid')(fc2_dropout)
        else:
            output = model.layers[-1](fc2_dropout)

        return Model(inputs=model.input, outputs=output)

    def get_model(self, data_params):
        K.clear_session()
        data_size = data_params['data_sizes'][0]
        net_model = self._conv_model(input_shape=data_size,
                                     include_top=True, weights=self._weights)
        net_model = self.add_dropout(net_model, add_new_pred=True)

        if self._freeze_conv_layers and self._weights:
            for layer in net_model.layers:
                if 'conv' in layer.name:
                    layer.trainable = False
        net_model.summary()
        return net_model

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.000005,
              optimizer='sgd',
              learning_scheduler=None,
              model_opts=None):

        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_opts['reshape'] = True
        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})
        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])

        # Get a copy of local parameters in the function minus self parameter
        local_params = {k: v for k, v in locals().items() if k != 'self'}

        #####  Optical flow model
        # Flow data shape: (1, num_frames, 224, 224, 2)
        self.train_model(model_type='opt_flow', **local_params)

        ##### rgb model
        self.train_model(model_type='rgb', **local_params)

        # Save settings
        model_opts_path, saved_files_path = get_path(**path_params,
                                                     file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)
        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        return saved_files_path


    def train_model(self, model_type, data_train, data_val,
                    class_w, learning_scheduler, path_params, optimizer,
                    batch_size, epochs, lr, **kwargs):
        """
        Trains a single model
        Args:
            train_data: Training data
            val_data: Validation data
            model_type: The model type, 'rgb' or 'opt_flow'
            path_params: Parameters for generating paths for saving models and configurations
            callbacks: List of training call back functions
            class_w: Class weights
            For other parameters refer to train()
        """
        learning_scheduler = learning_scheduler or {}
        if model_type == 'opt_flow':
            self._weights = None
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model = self.get_model(data_train['data_params_' + model_type])
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        data_path_params = {**path_params, 'sub_folder': model_type}
        model_path, _ = get_path(**data_path_params, file_name='model.h5')
        callbacks = self.get_callbacks(learning_scheduler,model_path)

        if data_val:
            data_val = data_val['data_' + model_type]
            if self._generator:
                data_val = data_val[0]

        history = train_model.fit(x=data_train['data_' + model_type][0],
                                  y=None if self._generator else data_train['data_' + model_type][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)

        if 'checkpoint' not in learning_scheduler:
            print('{} train model is saved to {}'.format(model_type, model_path))
            train_model.save(model_path)
        # Save training history
        history_path, saved_files_path = get_path(**data_path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

    def test(self, data_test, model_path=''):

        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        test_data = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})
        rgb_model = load_model(os.path.join(model_path, 'rgb', 'model.h5'))
        opt_flow_model = load_model(os.path.join(model_path, 'opt_flow', 'model.h5'))

        # Evaluate rgb model
        results_rgb = rgb_model.predict(test_data['data_rgb'][0], verbose=1)
        results_rgb = np.reshape(results_rgb, (-1, test_data['effective_dimension'], 1))
        results_rgb = np.mean(results_rgb, axis=1)

        # Evaluate optical flow model
        results_opt_flow = opt_flow_model.predict(test_data['data_opt_flow'][0], verbose=1)
        results_opt_flow = np.reshape(results_opt_flow, (-1, test_data['effective_dimension'], 1))
        results_opt_flow = np.mean(results_opt_flow, axis=1)

        # Average the predictions for both streams
        results = (results_rgb + results_opt_flow) / 2.0

        gt = np.reshape(test_data['data_rgb'][1], (-1, test_data['effective_dimension'], 1))[:, 1, :]

        acc = accuracy_score(gt, np.round(results))
        f1 = f1_score(gt, np.round(results))
        auc = roc_auc_score(gt, np.round(results))
        roc = roc_curve(gt, results)
        precision = precision_score(gt, np.round(results))
        recall = recall_score(gt, np.round(results))
        pre_recall = precision_recall_curve(gt, results)

        print('acc:{:.2f} auc:{:0.2f} f1:{:0.2f} precision:{:0.2f} recall:{:0.2f}'.format(acc, auc, f1, precision,
                                                                                          recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

        with open(save_results_path, 'w') as fid:
            yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


class TwoStreamFusion(ActionPredict):
    """
    This is an implementation of two-stream network with fusion mechanisms based
    on Feichtenhofer, Christoph et al. "Convolutional two-stream network fusion for
     video action recognition." CVPR, 2016.
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=True,
                 weights='imagenet',
                 fusion_point='early', # early, late, two-stage
                 fusion_method='sum',
                 num_classes=1,
                 backbone='vgg16',
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
            fusion_point: At what point the networks are fused (for details refer to the paper).
            Options are: 'early' (streams are fused after block 4),'late' (before the loss layer),
            'two-stage' (streams are fused after block 5 and before loss).
            fusion_method: How the weights of fused layers are combined.
            Options are: 'sum' (weights are summed), 'conv' (weights are concatenated and fed into
            a 1x1 conv to reduce dimensions to the original size).
            num_classes: Number of activity classes to predict.
            backbone: Backbone network. Only vgg16 is supported.
       """
        super().__init__(**kwargs)
        # Network parameters
        assert fusion_point in ['early', 'late', 'two-stage'], \
        "fusion point {} is not supported".format(fusion_point)

        assert fusion_method in ['sum', 'conv'], \
        "fusion method {} is not supported".format(fusion_method)

        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._num_classes = num_classes
        if backbone != 'vgg16':
            print("Only vgg16 backbone is supported")
        self._conv_models = vgg16.VGG16
        self._fusion_point = fusion_point
        self._fusion_method = fusion_method

    def get_data_sequence(self, data_type, data_raw, opts):
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'box': data_raw['bbox'].copy(),
             'crossing': data_raw['activities'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']

        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])

        else:
            overlap = opts['overlap'] if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res

            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs
            for seq in d['box']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
        for k in d.keys():
            d[k] = np.array(d[k])

        dcount = d['crossing'][:, 0, :]
        pos_count = np.count_nonzero(dcount)
        neg_count = len(dcount) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def get_data(self, data_type, data_raw, model_opts):
        model_opts['normalize_boxes'] = False
        process = False
        aux_name = '_'.join([self._backbone, 'raw']).strip('_')
        dataset = model_opts['dataset']
        eratio = model_opts['enlarge_ratio']
        self._generator = model_opts.get('generator', False)

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        feature_type = model_opts['obs_input_type'][0]

        # Only 3 types of rgb features are supported
        assert feature_type in ['local_box', 'local_context', 'scene']

        _data_samples = {'crossing': data['crossing']}
        data_type_sizes_dict = {}
        data_gen_params = {'data_type': data_type, 'crop_type': 'none'}

        if feature_type == 'local_box':
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif feature_type == 'local_context':
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio

        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')

        save_folder_name = '_'.join([feature_type, aux_name, str(eratio)]) \
                           if feature_type in ['local_context', 'local_surround'] \
                           else '_'.join([feature_type, aux_name])

        path_to_features, _ = get_path(save_folder=save_folder_name,
                                       dataset=dataset,
                                       save_root_folder='data/features')
        data_gen_params['save_path'] = path_to_features

        # Extract relevant rgb frames based on the optical flow length
        # Optical flow length is either 5 or 10. For example, for length of 10, and
        # sequence size of 16, 7 rgb frames are selected.
        ofl = model_opts['optical_flow_length']
        stidx = ofl - round((ofl + 1) / 2)
        endidx = (ofl + 1) // 2

        _data_samples['crossing'] = _data_samples['crossing'][:, stidx:-endidx, ...]
        effective_dimension = _data_samples['crossing'].shape[1]

        _data_samples[feature_type], feat_shape = self.load_images_crop_and_process(data['image'][:, stidx:-endidx, ...],
                                                                                    data['box_org'][:, stidx:-endidx, ...],
                                                                                    data['ped_id'][:, stidx:-endidx, ...],
                                                                                    process=process,
                                                                                    **data_gen_params)
        data_type_sizes_dict[feature_type] = feat_shape


        print('\n#####################################')
        print('Generating {} optical flow {}'.format(feature_type, data_type))
        print('#####################################')
        save_folder_name = '_'.join([feature_type, 'flow',  str(eratio)]) \
                                    if feature_type in ['local_context', 'local_surround'] \
                                    else '_'.join([feature_type, 'flow'])

        path_to_features, _ = get_path(save_folder=save_folder_name,
                                       dataset=dataset,
                                       save_root_folder='data/features')

        data_gen_params['save_path'] = path_to_features
        _data_samples['optical_flow'], feat_shape = self.get_optical_flow(data['image'],
                                                                          data['box_org'],
                                                                          data['ped_id'],
                                                                          **data_gen_params)

        # Create opflow data by stacking batches of optflow
        opt_flow = []
        if self._generator:
            _data_samples['optical_flow'] = np.expand_dims(_data_samples['optical_flow'], axis=-1)

        for sample in _data_samples['optical_flow']:
            opf = [np.concatenate(sample[i:i + ofl, ...], axis=-1) for i in range(sample.shape[0] - ofl + 1)]
            opt_flow.append(opf)
        _data_samples['optical_flow'] = np.array(opt_flow)
        if self._generator:
            data_type_sizes_dict['optical_flow'] = (feat_shape[0] - ofl + 1,
                                                    *feat_shape[1:3], feat_shape[3] * ofl)
        else:
            data_type_sizes_dict['optical_flow'] = _data_samples['optical_flow'].shape[1:]

        if model_opts.get('reshape', False):
            for k in _data_samples:
                dsize = data_type_sizes_dict.get(k, _data_samples[k].shape)
                if self._generator:
                    new_shape = (-1, _data_samples[k].shape[-1]) if _data_samples[k].ndim > 2 else (-1, 1)
                else:
                    new_shape = (-1,) + dsize[1:] if len(dsize) > 3 else (-1, dsize[-1])
                _data_samples[k] = np.reshape(_data_samples[k], new_shape)
                data_type_sizes_dict[k] = dsize[1:]

        _data = [_data_samples[feature_type], _data_samples['optical_flow']]
        data_sizes = [data_type_sizes_dict[feature_type],
                      data_type_sizes_dict['optical_flow']]
        data_types = [feature_type, 'optical_flow']

        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=_data_samples['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=[feature_type,'optical_flow'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test',
                                   stack_feats=True),
                                   _data_samples['crossing'])
        else:
            _data = (_data, _data_samples['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types,
                                'data_sizes': data_sizes},
                'effective_dimension': effective_dimension,
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def fuse_layers(self, l1, l2):
        """
        Fuses two given layers (tensors)
        Args:
            l1:  First tensor
            l2:  Second tensor
        Returns:
            Fused tensors based on a given method.
        """

        if self._fusion_method == 'sum':
            return Add()([l1, l2])
        elif self._fusion_method == 'conv':
            concat_layer = Concatenate()([l1, l2])
            return Conv2D(l2.shape[-1], 1, 1)(concat_layer)

    def add_dropout(self, model, add_new_pred = False):
        """
        Adds dropout layers to a given vgg16 network. If specified, changes the dimension of
        the last layer (predictions)
        Args:
            model: A given vgg16 model
            add_new_pred: Whether to change the final layer
        Returns:
            Returns the new model
        """

        # Change to a single class output and add dropout
        fc1_dropout = Dropout(self._dropout)(model.layers[-3].output)
        fc2 = model.layers[-2](fc1_dropout)
        fc2_dropout = Dropout(self._dropout)(fc2)
        if add_new_pred:
            output = Dense(self._num_classes, name='predictions', activation='sigmoid')(fc2_dropout)
        else:
            output = model.layers[-1](fc2_dropout)

        return Model(inputs=model.input, outputs=output)

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.000005,
              optimizer='sgd',
              learning_scheduler=None,
              model_opts=None):
        learning_scheduler = learning_scheduler or {}

        # Generate parameters for saving models and configurations
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')

        model_opts['reshape'] = True
        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})
            data_val = data_val['data']
            if self._generator:
                data_val = data_val[0]

        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model = self.get_model(data_train['data_params'])
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        callbacks = self.get_callbacks(learning_scheduler, model_path)

        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)
        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        # Save settings
        model_opts_path, saved_files_path = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)
        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        return saved_files_path

    def get_model(self, data_params):
        data_size = data_params['data_sizes'][0]
        rgb_model = self._conv_models(input_shape=data_size,
                                          include_top=True, weights=self._weights)
        rgb_model = self.add_dropout(rgb_model, add_new_pred=True)
        data_size = data_params['data_sizes'][1]
        temporal_model = self._conv_models(input_shape=data_size,
                                           include_top=True, weights=None, classes=1)
        temporal_model = self.add_dropout(temporal_model)

        for layer in rgb_model.layers:
            layer._name = "rgb_" + layer._name
            if self._freeze_conv_layers and 'conv' in layer._name:
                layer.trainable = False

        # rgb_model.load_weights('')
        if self._fusion_point == 'late':
            output = Average()([temporal_model.output, rgb_model.output])

        if self._fusion_point == 'early':
            fusion_point = 'block4_pool'
            rgb_fuse_layer = rgb_model.get_layer('rgb_' + fusion_point).output
            start_fusion = False
            for layer in temporal_model.layers:
                if layer.name == fusion_point:
                    x = self.fuse_layers(rgb_fuse_layer, layer.output)
                    start_fusion = True
                    continue
                if start_fusion:
                    x = layer(x)
                else:
                   layer.trainable = False

            output = x

        if self._fusion_point == 'two-stage':
            fusion_point = 'block5_conv3'
            rgb_fuse_layer = rgb_model.get_layer('rgb_' + fusion_point).output
            start_fusion = False
            for layer in temporal_model.layers:
                if layer.name == fusion_point:
                    x = self.fuse_layers(rgb_fuse_layer, layer.output)
                    start_fusion = True
                    continue
                if start_fusion:
                    x = layer(x)
                else:
                   layer.trainable = False

            output = Average()([x, rgb_model.output])

        net_model = Model(inputs=[rgb_model.input, temporal_model.input],
                          outputs=output)
        plot_model(net_model, to_file='model.png',
                   show_shapes=False, show_layer_names=False,
                   rankdir='TB', expand_nested=False, dpi=96)

        net_model.summary()
        return net_model

    def test(self, data_test, model_path=''):
        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        data_test = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})

        # Load conv model
        test_model = load_model(os.path.join(model_path, 'model.h5'))
        results = test_model.predict(data_test['data'][0], batch_size=8, verbose=1)
        results = np.reshape(results, (-1, data_test['effective_dimension'], 1))
        results = np.mean(results, axis=1)

        gt = np.reshape(data_test['data'][1], (-1, data_test['effective_dimension'], 1))[:, 1, :]
        acc = accuracy_score(gt, np.round(results))
        f1 = f1_score(gt, np.round(results))
        auc = roc_auc_score(gt, np.round(results))
        roc = roc_curve(gt, results)
        precision = precision_score(gt, np.round(results))
        recall = recall_score(gt, np.round(results))
        pre_recall = precision_recall_curve(gt, results)

        print('acc:{:.2f} auc:{:0.2f} f1:{:0.2f} precision:{:0.2f} recall:{:0.2f}'.format(acc, auc, f1, precision,
                                                                                  recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

        with open(save_results_path, 'w') as fid:
            yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


def attention_3d_block(hidden_states, dense_size=128, modality=''):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    @author: felixhao28.
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec'+modality)(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state'+modality)(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score'+modality)
    attention_weights = Activation('softmax', name='attention_weight'+modality)(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector'+modality)
    pre_activation = concatenate([context_vector, h_t], name='attention_output'+modality)
    attention_vector = Dense(dense_size, use_bias=False, activation='tanh', name='attention_vector'+modality)(pre_activation)
    return attention_vector


class PCPA(ActionPredict):

    """
    Hybridization of MultiRNN with 3D convolutional features and attention

    many-to-one attention block is adapted from:
    https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention/attention.py

    """
    
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function
        
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='../data/features')  ##change
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing']) # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        conv3d_model = self._3dconv()
        network_inputs.append(conv3d_model.input)

        attention_size = self._num_hidden_units

        if self._backbone == 'i3d':
            x = Flatten(name='flatten_output')(conv3d_model.output)
            x = Dense(name='emb_'+self._backbone,
                       units=attention_size,
                       activation='sigmoid')(x)
        else:
            x = conv3d_model.output
            x = Dense(name='emb_'+self._backbone,
                       units=attention_size,
                       activation='sigmoid')(x)

        encoder_outputs.append(x)

        for i in range(1, core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(network_inputs[i]))

        if len(encoder_outputs) > 1:
            att_enc_out = []
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(encoder_outputs[0])
            att_enc_out.append(x) # first output is from 3d conv netwrok 
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[1:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_'+data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

            #print(encodings.shape)
            #print(weights_softmax.shape)
        else:
            encodings = encoder_outputs[0]

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        net_model.summary()
        #plot_model(net_model, to_file='MultiRNN3D_ATT.png')
        return net_model

class PIE_TFGRU(ActionPredict):
    """
    Pedestrian crossing prediction based on
    Rasouli et al. "Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs."
    BMVC, 2020. The original code can be found at https://github.com/aras62/SF-GRU
    """
    def __init__(self,
                 num_hidden_units=128,  ### orig 256
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        return_sequence = True
        num_layers = len(data_sizes)


        y = []
        ear_fusion = []

        for i in range(num_layers):

            print("#################################")
            print(data_types[i])
            print("data_sizes:",data_sizes[i])

            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

             ####later fusion all in one (best)
            if (data_types[i] == 'local_box') | (data_types[i] == 'local_context') | (data_types[i] == 'local_surround'):
                feature_depth = 128
                d_point = 128
                enc_count = 1
                att_head_count = 4
            
            elif (data_types[i] == 'local_box_img') | (data_types[i] == 'local_context_img'):
                feature_depth = 128
                d_point = 128
                enc_count = 1
                att_head_count = 4

            elif (data_types[i] == 'pose'):
                feature_depth = 128
                d_point = 64
                enc_count = 2
                att_head_count = 4
                
            elif (data_types[i] == 'headpose'):
                feature_depth = 128
                d_point = 64
                enc_count = 1
                att_head_count = 4

            elif (data_types[i] == 'box'):
                feature_depth = 128
                d_point = 128  ##512
                enc_count = 1
                att_head_count = 4
            
            elif (data_types[i] == 'speed'):
                feature_depth = 128
                d_point = 128
                enc_count = 1
                att_head_count = 4
            
            elif (data_types[i] == 'TSC'):
                feature_depth = 128
                d_point = 128
                enc_count = 1
                att_head_count = 4
            

            #  #################################CHN ATT#################################
            # if 0 <= i < (num_layers - 1):
            #     y.append(tf.expand_dims(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i]), axis=2))
            # else:
            #     y.append(tf.expand_dims(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i]), axis=2))

            #     xy = Concatenate(axis=2)([i for i in y])

            #     xy = tf.transpose(xy, perm=[0, 1, 3, 2])

            #     Fusion_ATT = channel_attention()(xy)###
            #     xy = Fusion_ATT ####

            #     xy = tf.transpose(xy, perm=[0, 2, 3, 1])
                    
            #     xy = Conv1D(1, 1)(xy)

            #     xy = tf.squeeze(xy,[-1])

            #     xy = Conv1D(1, 1)(xy)   ###

            #     xy = tf.squeeze(xy,[-1])   ###


                # xy = tf.keras.layers.Flatten()(xy)
            
            #########################################################################
            
            #################################main method#################################
            if 0 <= i < (num_layers - 1):
                y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i]))
            else:
                y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i]))

                xy = Concatenate(axis=2)([i for i in y])

                xy = tf.transpose(xy, perm=[0, 2, 1])

                ##ch att###
                # Fusion_ATT = channel_attention()(xy)###
                # xy = Fusion_ATT ####

                # xy = tf.transpose(xy, perm=[0, 2, 3, 1])
                    
                # xy = Conv1D(1, 1)(xy)

                # xy = tf.squeeze(xy,[-1])
                #####

                xy = Conv1D(1, 1)(xy)   ###

                xy = tf.squeeze(xy,[-1])   ###
           
                # xy = Dense(128, activation=None, name='fusion_layer', activity_regularizer=regularizers.l2(0.001))(xy)
                xy = Dense(128, activation=None, name='fusion_layer')(xy)

          
            #######################################################

            #################################GRU main method#################################
            # if 0 <= i < (num_layers - 1):
            #     y.append(self._gru(name='enc_' + data_types[i], r_sequence=False)(network_inputs[i]))
            # else:
            #     y.append(self._gru(name='enc_' + data_types[i], r_sequence=False)(network_inputs[i]))

            #     xy = Concatenate(axis=1)([i for i in y])

                

            #     ##ch att###
            #     # Fusion_ATT = channel_attention()(xy)###
            #     # xy = Fusion_ATT ####

            #     # xy = tf.transpose(xy, perm=[0, 2, 3, 1])
                    
            #     # xy = Conv1D(1, 1)(xy)

            #     # xy = tf.squeeze(xy,[-1])
            #     #####

           
            #     xy = Dense(128, activation=None, name='fusion_layer')(xy)

          
            #######################################################


            #########for only one data ablation#################################           
            # if 0 <= i < (num_layers - 1):
            #     y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i]))
            # else:
            #     y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i]))

            #     # xy = Concatenate(axis=2)([i for i in y])

            #     y = y[0]


            #     xy = tf.transpose(y, perm=[0, 2, 1])

            #     # Fusion_ATT = channel_attention()(xy)###
            #     # xy = Fusion_ATT ####

            #     # xy = tf.transpose(xy, perm=[0, 2, 3, 1])
                    
            #     # xy = Conv1D(1, 1)(xy)

            #     # xy = tf.squeeze(xy,[-1])

            #     xy = Conv1D(1, 1)(xy)   ###

            #     xy = tf.squeeze(xy,[-1])   ###

            #     # xy = Dense(32, activation=None, name='fusion_layer')(xy)

          
            #################################################
            
            #############early fusion
            # if i == 0:
            #     x = self._Trans(name='Trans_Enc_' + data_types[i])(network_inputs[i])
          
            #     # print(type(x.numpy()))

                
     

            #     # tf.compat.v1.disable_eager_execution()
            #     # with tf.compat.v1.Session() as sess:
            #     #     print(sess.run(x))
            #     # x= self.nothing(x)
            #     # x = tf.py_function(func = self.nothing, inp = x, Tout = tf.float32)

            #     # tf.config.run_functions_eagerly(True)
            #     # tf.compat.v1.enable_eager_execution()
            #     # tf.enable_eager_execution() 
  
            #     # self.visulize_attention_ratio('/datadisk/PIE/Research/PIEPredict-master/PIE_dataset/images/set01/video_0002/04502.png', x)
               
            # # elif i == 1:
            # #     x = Concatenate(axis=2)([x, network_inputs[i]])

            # #     x = self._Trans(name='Trans_Enc_' + data_types[i])(x)
            # elif i == 1:
            #     x2 = self._Trans(name='Trans_Enc_' + data_types[i])(network_inputs[i]) 

            #     x = Concatenate(axis=2)([x, x2])
            # elif 2 <= i < (num_layers - 1):
            #     y.append((network_inputs[i]))
            # else:
               
            #     y.append((network_inputs[i]))
            #     # print(y)
            #     yCon = Concatenate(axis=2)([i for i in y])
            #     yCon = Dense(256, activation=None, name='fusion_layer_0')(yCon)

            #     yCon = (self._Trans(name='Trans_Enc_' + data_types[i])(yCon))

            #     # print("3333sd")
            #     # print(y)
               


            #     xy = Concatenate(axis=2)([x, yCon]) #########here

            #     xy = tf.transpose(a=xy, perm=[0, 2, 1])
                
            #     xy = Conv1D(1, 1)(xy)

            #     xy = tf.squeeze(xy,[-1])

            #     # print("this is xy output:")
            #     # print(xy.shape)

            #     # xy = Dense(256, activation=None, name='fusion_layer')(xy)
        #    #######################################################

        ####test hiarachical TF#######################################
            # if i == 0:
            #     x = self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i])

            # elif 0 < i < (num_layers - 1):
            #     x = Concatenate(axis=2)([x, network_inputs[i]])

            #     x = self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(x)
            # else:
            #     x = Concatenate(axis=2)([x, network_inputs[i]])

            #     x = self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(x)
        

            #     xy = tf.transpose(x, perm=[0, 2, 1])

            #     xy = Conv1D(1, 1)(xy)
            #     # print("this is no POOLING x output:")
            #     # print(x.shape)
                    
            #     # xy = tf.keras.layers.GlobalAveragePooling1D()(x)
                
            #     # print("this is xy output:")
            #     # print(xy.shape)

            #     xy = tf.squeeze(xy,[-1])

            #     ####test for vit
            #     # xy = Concatenate(axis=-1)([xy, y[0], y[1]])


            #     # print("this is xy output:")
            #     # print(xy.shape)


            #     # xy = Dense(64, activation=None, name='fusion_layer')(xy)
        ##################################################################

            #################################Test mix the speed and TSC #################################
        #     if (data_types[i] == 'TSC'):
        #         continue
        #     elif (data_types[i] == 'speed'):
        #         ear_fusion = Concatenate(axis=2)([network_inputs[l] for l in range(num_layers) if ((data_types[l] == 'speed') or (data_types[l] == 'TSC'))])
        #         y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(ear_fusion))
        #     else:
        #         y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i]))
        #     # if (data_types[i] == 'headpose'):
        #     #     continue
        #     # elif (data_types[i] == 'pose'):
        #     #     ear_fusion1 = Concatenate(axis=2)([network_inputs[l] for l in range(num_layers) if ((data_types[l] == 'pose') or (data_types[l] == 'headpose'))])
        #     #     y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(ear_fusion1))  #change for concat the pose and headpose
        #     # elif (data_types[i] == 'TSC'):
        #     #     continue
        #     # elif (data_types[i] == 'speed'):
        #     #     ear_fusion2 = Concatenate(axis=2)([network_inputs[l] for l in range(num_layers) if ((data_types[l] == 'speed') or (data_types[l] == 'TSC'))])
        #     #     y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(ear_fusion2))  #change for concat speed and TSC
        #     # else:
        #     #     y.append(self._Trans(name='Trans_Enc_' + data_types[i], d_model = feature_depth, d_point_wise_ff = d_point, encoder_count = enc_count, attention_head_count = att_head_count)(network_inputs[i]))

        # xy = Concatenate(axis=2)([i for i in y])

        # xy = tf.transpose(xy, perm=[0, 2, 1])

        # # Fusion_ATT = channel_attention()(xy)###
        # # xy = Fusion_ATT ####

        # # xy = tf.transpose(xy, perm=[0, 2, 3, 1])
            
        # # xy = Conv1D(1, 1)(xy)

        # # xy = tf.squeeze(xy,[-1])

        # xy = Conv1D(1, 1)(xy)   ###

        # xy = tf.squeeze(xy,[-1])   ###


           
              #######################################################
        

        # model_output = Dense(1, activation='sigmoid', name='output_dense', activity_regularizer=regularizers.l2(0.001))(xy)
        model_output = Dense(1, activation='sigmoid', name='output_dense')(xy)

        net_model = Model(inputs=network_inputs, outputs=model_output)

        return net_model

    def _Trans(self, name = 'Transformer', inputs_vocab_size = 32000,
                 output_size = 3200,
                 encoder_count = 1,
                 attention_head_count = 8,
                 d_model = 256,
                 d_point_wise_ff = 1024,
                 dropout_prob = 0.1):
        """
        A helper function to creat a Transformer Encoder layer
        :param name: Name of the layer
        :param r_state: Whether to return the states of the GRU
        :param r_sequence: Whether to return sequence
        :return: A GRU unit
        """


        return Transformer_Encoder(inputs_vocab_size,
                                    output_size,
                                    encoder_count,
                                    attention_head_count,
                                    d_model,
                                    d_point_wise_ff,
                                    dropout_prob)



class PIE_TF(ActionPredict):

    """
    Hybridization of MultiRNN with 3D convolutional features and attention

    many-to-one attention block is adapted from:
    https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention/attention.py

    """
    
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function
        
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing']) # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        conv3d_model = self._3dconv()
        network_inputs.append(conv3d_model.input)

        attention_size = self._num_hidden_units

        if self._backbone == 'i3d':
            x = Flatten(name='flatten_output')(conv3d_model.output)
            x = Dense(name='emb_'+self._backbone,
                       units=attention_size,
                       activation='sigmoid')(x)
        else:
            x = conv3d_model.output
            x = Dense(name='emb_'+self._backbone,
                       units=attention_size,
                       activation='sigmoid')(x)

        encoder_outputs.append(x)

        for i in range(1, core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(network_inputs[i]))

        if len(encoder_outputs) > 1:
            att_enc_out = []
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(encoder_outputs[0])
            att_enc_out.append(x) # first output is from 3d conv netwrok 
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[1:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_'+data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

            #print(encodings.shape)
            #print(weights_softmax.shape)
        else:
            encodings = encoder_outputs[0]

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        net_model.summary()
        #plot_model(net_model, to_file='MultiRNN3D_ATT.png')
        return net_model




def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))


class DataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 labels=None,
                 data_sizes=None,
                 process=False,
                 global_pooling=None,
                 input_type_list=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 stack_feats=False):
        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list
        self.batch_size = 1 if len(self.labels) < batch_size  else batch_size        
        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data[0])/self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]

        X = self._generate_X(indices)
        if self.to_fit:
            y = self._generate_y(indices)
            return X, y
        else:
            return X

    def _get_img_features(self, cached_path):
        # print(cached_path)
        with open(cached_path, 'rb') as fid:
            try:
                img_features = pickle.load(fid)
            except:
                img_features = pickle.load(fid, encoding='bytes')
        if self.process:
            if self.global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self.global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()        
        return img_features

    def _generate_X(self, indices):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            num_ch = features_batch.shape[-1]//len(self.data[input_type_idx][0])
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        if 'flow' in input_type:
                            img_features = read_flow_file(cached_path)
                        else:
                            img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            # for static model if only one image in the sequence
                            features_batch[i, ] = img_features
                        else:
                            if self.stack_feats and 'flow' in input_type:
                                features_batch[i,...,j*num_ch:j*num_ch+num_ch] = img_features
                            else:
                                features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices):
        return np.array(self.labels[indices])

