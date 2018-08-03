import numpy as np
# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.preprocessing.alan import ALANGenerator
from keras_retinanet.utils.eval import jg_evaluate
from keras_retinanet.utils.keras_version import check_keras_version


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

from keras_retinanet.preprocessing.alan import TestSequence, mkdir_p, prediction_saver, non_max_suppression_fast_score, evaluate_csv
from keras_retinanet.utils.visualization import draw_detections
from keras.utils import Sequence, OrderedEnqueuer
from tqdm import tqdm

import numpy as np
import os

import cv2

def label_to_name(label):
    if label==0:
        return "etiquette"
    else:
        return str(label)

# # hard examples for Albert V1:
# filenames_to_test = ["CAT_1_9804",
# "CAT_1_9091",
# "CAT_1_8021",
# "CAT_1_8023",
# "CAT_1_9844",
# "CAT_1_7012",
# "CAT_1_7013",
# "CAT_1_7014",
# "CAT_1_9035",
# "CAT_1_11019",
# "CAT_1_9048",
# "CAT_1_10052",
# "CAT_1_9803"]

# SAL1 with available and checked groundtruth:
filenames_to_test = ["PANO_SAL1_BR_5026",
"PANO_SAL1_BR_6201",
"PANO_SAL1_BR_6300",
"PANO_SAL1_BR_6516",
"PANO_SAL1_BR_7003",
"PANO_SAL1_BR_7004",
"PANO_SAL1_BR_7206",
"PANO_SAL1_BR_8008",
"PANO_SAL1_BR_8022",
"PANO_SAL1_BR_9023",]

def get_niveau(filename):
    num = filename.split('_')[-1]
    if len(num)==4:
        niveau = num[:1]
    elif len(num)==5:
        niveau = num[:2]
    else:
        print("num",num)
    return niveau
def get_tranche(filename):
    if "PANO" in filename:
        return filename.split('_')[1]+filename.split('_')[1]
    else:
        return filename.split('_')[0]+filename.split('_')[1]



print('Loading model, this may take a second...')
# model_path = "/dds/work/workspace/alan_tmp_files/resnet50_alan_01_01aout_19h15_inference.h5"
# model_path =  "/dds/work/workspace/keras-retinanet/snapshots/alan_training_02_1eraout/resnet50_alan_13_inference.h5"
# model_path =  "/dds/work/workspace/keras-retinanet/snapshots/alan_training_02_1eraout/resnet50_alan_02_inference.h5"
model_path = "/dds/work/workspace/keras-retinanet/snapshots/alan_training_02_1eraout/resnet50_alan_06_inference.h5"
model = models.load_model(model_path, backbone_name='resnet50')


score_threshold = 0.1
max_detections = 100

save_path = "/dds/work/workspace/alan_tmp_files/results_apply_alan/multiple_files"
psaver_all = prediction_saver(save_path,"multiple_files")

for filename in tqdm(filenames_to_test, desc="files to test",ncols=len(filenames_to_test)):
    image_to_test = "/dds/work/workspace/alan_jpg_files/"+get_tranche(filename)+"/Niveau "+get_niveau(filename)+"/"+filename+".jpg"
    print(image_to_test,os.path.isfile(image_to_test))


    save_path = "/dds/work/workspace/alan_tmp_files/results_apply_alan/"+filename
    if save_path is not None:
        mkdir_p(save_path)
    data_sequence = TestSequence(image_path=image_to_test, folder_crops="/dds/work/workspace/alan_tmp_files/sequence_crops")
    psaver = prediction_saver(save_path,filename)
    ordered_data_sequence = OrderedEnqueuer(data_sequence, use_multiprocessing=True)
    ordered_data_sequence.start(workers=4, max_queue_size=100)
    datas = ordered_data_sequence.get()
    t = tqdm(datas,total=len(data_sequence))

    for id, xywh, image in t:
        if len([None for x in data_sequence.results if x is None])==0:
            break
        if data_sequence.results[id] is not None:
            continue
        # if not (id==211 or id==245):
        #     continue

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]



        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]

        indices_postnms = non_max_suppression_fast_score(image_boxes, 0.1, image_scores)
        # print(40 * "#", id)
        # print("indices_postnms", indices_postnms)
        # print("image_boxes",image_boxes.shape,image_boxes)
        # print("image_scores",image_scores.shape,image_scores)
        # print("image_labels",image_labels.shape,image_labels)

        image_boxes      = image_boxes[indices_postnms]
        image_scores     = image_scores[indices_postnms]
        image_labels     = image_labels[indices_postnms]

        #
        # print("image_boxes",image_boxes.shape,image_boxes)
        # print("image_scores",image_scores.shape,image_scores)
        # print("image_labels",image_labels.shape,image_labels)
        # print("indices_postnms", indices_postnms)


        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        data_sequence.results[id]=image_detections
        if save_path is not None and len(indices)>0:
            draw_detections(image, image_boxes, image_scores, image_labels, label_to_name=label_to_name,score_threshold=0)
            cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(id)), image)

        # if len(indices)>0:
        psaver.add_instances(data_sequence.filename, xywh, image_boxes, image_scores)
        psaver_all.add_instances(data_sequence.filename, xywh, image_boxes, image_scores)
    psaver.save()
    psaver.save_nms()
    psaver.save_tagbrowser()
    psaver.save_nms_tagbrowser()
psaver_all.save()
csv_pred_all = psaver_all.save_nms()
psaver_all.save_nms_tagbrowser()
psaver_all.save_tagbrowser()

# with open(saving_result_path, 'wb') as handle:
#     pickle.dump(img_infos, handle, protocol = pickle.HIGHEST_PROTOCOL)
# ordered_data_sequence.stop()


paths_csv_groundtruth = ["/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_5026.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_6201.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_6300.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_6516.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_7003.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_7004.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_7206.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_8008.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_8022.csv",
"/dds/work/workspace/alan_jpg_files/SAL1_eval/SAL1BR_groundtruth-PANO_SAL1_BR_9023.csv",]

paths_csv_detection = [csv_pred_all]
evaluate_csv(paths_csv_detection, paths_csv_groundtruth, iou_threshold=0.0005, score_threshold=score_threshold, debug = True)
