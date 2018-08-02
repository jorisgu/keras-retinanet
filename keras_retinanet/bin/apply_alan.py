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

from keras_retinanet.preprocessing.alan import TestSequence, mkdir_p, prediction_saver, non_max_suppression_fast_score
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



image_to_test = "/dds/work/workspace/alan_jpg_files/SAL1/Niveau 6/PANO_SAL1_BR_6201.jpg"
image_to_test = "/dds/work/workspace/alan_jpg_files/SAL1/Niveau 7/PANO_SAL1_BR_7003.jpg"

print('Loading model, this may take a second...')
# model_path = "/dds/work/workspace/alan_tmp_files/resnet50_alan_01_01aout_19h15_inference.h5"
# model_path =  "/dds/work/workspace/keras-retinanet/snapshots/alan_training_02_1eraout/resnet50_alan_13_inference.h5"
# model_path =  "/dds/work/workspace/keras-retinanet/snapshots/alan_training_02_1eraout/resnet50_alan_02_inference.h5"
model_path = "/dds/work/workspace/keras-retinanet/snapshots/alan_training_02_1eraout/resnet50_alan_06_inference.h5"
model = models.load_model(model_path, backbone_name='resnet50')


score_threshold = 0.1
max_detections = 100


save_path = "/dds/work/workspace/alan_tmp_files/results_apply_alan"
if save_path is not None:
    mkdir_p(save_path)
data_sequence = TestSequence(image_path=image_to_test, folder_crops="/dds/work/workspace/alan_tmp_files/sequence_crops")
psaver = prediction_saver(save_path)
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
psaver.save()
psaver.save_nms()

# with open(saving_result_path, 'wb') as handle:
#     pickle.dump(img_infos, handle, protocol = pickle.HIGHEST_PROTOCOL)
ordered_data_sequence.stop()
