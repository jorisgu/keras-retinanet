# git pull; CUDA_VISIBLE_DEVICES=1 python keras_retinanet/bin/train.py --freeze-backbone --no-evaluation --batch-size=1 --num-workers=12 --random-transform alan
from .generator import Generator
from keras.utils import Sequence, OrderedEnqueuer
from ..utils.compute_overlap import compute_overlap
import os
from os import path as osp
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = 1e10
import errno
import numpy as np
import glob
from tqdm import tqdm
import pickle
from math import ceil, floor
import csv
import ntpath
import random
import time

# Malisiewicz et al.
def non_max_suppression_fast_score(boxes, overlapThresh, scores):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick #boxes[pick].astype("int")

def change_tile(tile, new_width, new_height, memory_offset):
    tup = tile[0]
    return [(tup[0],) + ((0,0,new_width, new_height),) + (tup[-2]+memory_offset,) + (tup[-1],)]

def read_line_portion(img_path,x,y,w,h,i):
    img_pil = Image.open(img_path)
    W = img_pil.size[0]
    img_pil.size=(w,1)
    memory_offset = (x+i)*3*W+3*y
    img_pil.tile = change_tile(img_pil.tile,w,1,memory_offset)
    #print(img_pil.tile)
    #print(img_pil.size)
    return img_pil

def read_from_memory(img_path,x,y,w,h):
    result = Image.new('RGB',(w,h))
    try:
        for i in range(h):
            a = read_line_portion(img_path, x,y,w,h,i)
            result.paste(a,(0,i))
        return result
    except:
        print("Error with file:", img_path)

def pil_resize(img_pil, new_max_size = 1000.):
    size = img_pil.size
    max_size_img = float(max(size))
    new_size =  tuple((new_max_size/max_size_img*np.asarray(size)).astype(int))
    img_pil_resized = img_pil.resize(new_size, Image.NEAREST)
    return img_pil_resized

def make_spk(W, w, overlap, scale):
    """ input :
        W : dimension de l'image
        w : dimension du crop
        overlap : recouvrement dans le décalage des crops avec strategie de sliding window
        scale : zoom

        output :
        sp (=s') : recalcul du stride s pour améliorer le recouvrement de l'image
        k : nombre de crop que l'on peut faire le long de la dimension considérée

        Dans le principe, avec le stride de base s le crop k sort de l'image,
        donc on diminue légèrement le stride pour que le k-ième crop arrive dans l'image.

        La conséquence directe est que l'overlap augmente un peu.
        """
    wp=ceil(w*scale) #pour s'assurer d'avoir des pixels
    s=floor(wp-overlap*wp) #floor pour faciliter la redondance d'info
    #w'+k*s=W
    k=ceil((W-wp)/s) #ceil pour faciliter la redondance on s'assure de dépasser un peu de l'image

    #Exemple : k=ceil((15000-1125)/(1125-0.5*1125))
    #                      = 13875/562 = ceil(24,68)=25
    #Erreur sur W:
    #|W-w'-k*s|=-175
    #recalcul du s pour resserrer les crops
    sp=floor((W-wp)/k)  #=555
    #recalcul de l'erreur sur W
    #W-w'-k*s'=0
    #     key='D'+str(W)+'_d'+str(w)+'_o'+str(overlap)+'_s'+str(scale)
    return (sp,k)

def load_crop(xywh,img_path,dim=1000):
    try:
        # xywh = crop[2]
        if img_path[-3:]=='ppm':
            return read_from_memory(img_path,xywh[0],xywh[1],xywh[2],xywh[3])
        else:
        #     # print("Loading crop directly from source ["+img_path+"] with xywh",xywh,".")

            img = Image.open(img_path)
            return img.crop((xywh[1],xywh[0],xywh[1]+xywh[2],xywh[0]+xywh[3]))
    except:
        print("Error in load_crop with file name:",img_path, crop)

class mask:
    """ The objectif of the mask is to gather information about a pano in list format (ie. low memory consumption)
    and to be able to generate an area preview at a specific scale to work on it."""
    def __init__(self, filepath, obj_sources = [], validation_sources = [], filter=True):
        self.filepath = filepath
        self.objs = obj_sources
        self.validations = validation_sources
        self.filter = filter
        # print("self filter", self.filter)

    def produce_gt(self, crop_in, debug=False):
        """ to produce fake mask for training on background """

        size_out = (crop_in[3],crop_in[2]) #(xywh)
                    # (id_obj,(new_x1,new_y1,new_x2,new_y2),classe, score)
        objs = [  (-1,(1,1,300,300),0, 1),
                    (-1,(200,200,500,500),0, 1),
                    (-1,(1,200,300,500),0, 1),
                    (-1,(200,1,500,300),0, 1),
                    (-1,(100,100,400,400),0, 1)]
        gt_np = np.zeros(size_out+(len(objs),))
        # print("gt_np shape",gt_np.shape)
        for id,(obj_id,obj_intersection,classe,score) in enumerate(objs):
            gt_np[obj_intersection[1]:obj_intersection[3],obj_intersection[0]:obj_intersection[2],id]=1
        return gt_np

    def get_gt(self, crop_in, debug=False):
        size_out = (crop_in[3],crop_in[2]) #(xywh)
        objs = self.get_objs_in_crop(crop_in,debug=debug)
        gt_np = np.zeros(size_out+(len(objs),))
        # print("gt_np shape",gt_np.shape)
        for id,(obj_id,obj_intersection,classe,score) in enumerate(objs):
            gt_np[obj_intersection[1]:obj_intersection[3],obj_intersection[0]:obj_intersection[2],id]=1
        return gt_np

    def get_gt_lm(self, crop_in, debug=False):
        """ get groundtruth low memory"""
        size_out = (crop_in[3],crop_in[2]) #(xywh)
        objs = self.get_objs_in_crop(crop_in,debug=debug)
        boxes = [l[1] for l in objs]
        classes = [l[2] for l in objs]
        scores = [l[3] for l in objs]
        return boxes,classes,scores

    def convert_polygon_to_rectangle(self, points):
        """points = [(x,y),(x,y),(x,y)...]"""
        x1=points[0][0]
        y1=points[0][1]
        x2=points[0][0]
        y2=points[0][1]
        for point in points:
            if point[0]<x1:
                x1=point[0]
            if point[0]>x2:
                x2=point[0]
            if point[1]<y1:
                y1=point[1]
            if point[1]>y2:
                y2=point[1]
        return (x1,y1,x2,y2)

    def get_validations_in_crop(self, crop):
        validations = []
        for id_validation,validation in enumerate(self.validations):
            if int(validation['valid'])>0:
                classe = 'valid'
            elif int(validation['invalid'])>0:
                classe = 'invalid'
            else:
                classe = 'unspecified'

            (x1,y1,x2,y2) = (int(validation["x1"]),int(validation["y1"]),int(validation["x2"]),int(validation["y2"]))
            new_x1 = max(x1, crop[1])-crop[1]
            new_y1 = max(y1, crop[0])-crop[0]
            new_x2 = min(x2, crop[1]+crop[2])-crop[1]
            new_y2 = min(y2, crop[0]+crop[3])-crop[0]
            if new_x1 > new_x2 or new_y1 > new_y2:
                pass
            else:
                validations.append((id_validation,(new_x1,new_y1,new_x2,new_y2),classe))
        return validations

    def get_objs_in_crop(self, crop, debug=False):
        objs = []
        for id_obj,obj in enumerate(self.objs):
            if len(obj['equipment'])>0:
                classe = 0
            elif len(obj['extra_equipment'])>0:
                classe = 0
            else:
                continue
            # elif len(obj['hard_background'])>0:
            #     classe =
            # else:
            #     classe = 0
                # continue
            if "score" in obj.keys():
                score=float(obj['score'])
            else:
                score=None
            (x1,y1,x2,y2) = self.convert_polygon_to_rectangle(
            [(int(obj["x1"]),int(obj["y1"])),
            (int(obj["x2"]),int(obj["y2"])),
            (int(obj["x3"]),int(obj["y3"])),
            (int(obj["x4"]),int(obj["y4"]))])

            new_x1 = max(x1, crop[1])-crop[1]
            new_y1 = max(y1, crop[0])-crop[0]
            new_x2 = min(x2, crop[1]+crop[2])-crop[1]
            new_y2 = min(y2, crop[0]+crop[3])-crop[0]
            if new_x1 > new_x2 or new_y1 > new_y2:
                # intersection empty
                pass
            elif self.filter:
                if new_x1==0 or new_y1==0 or new_x2==crop[2] or new_y2==crop[3]:
                    pass
                else:
                    objs.append((id_obj, (new_x1,new_y1,new_x2,new_y2), classe, score))
                    if debug:
                        print("filter")
                        print("         crop:",crop)
                        print("obj in crop:",(new_x1,new_y1,new_x2,new_y2))
                        print("        score:",score)
            else:
                objs.append((id_obj,(new_x1,new_y1,new_x2,new_y2),classe, score))
                if debug:
                    print("no filter")
                    print("         crop:",crop)
                    print("obj in crop:",(new_x1,new_y1,new_x2,new_y2))
                    print("        score:",score)
        return objs

    def contains_obj(self,crop):
        objs = []
        for id_obj,obj in enumerate(self.objs):
            if len(obj['equipment'])>0:
                classe = 0
            elif len(obj['extra_equipment'])>0:
                classe = 0
            else:
                continue
            # elif len(obj['hard_background'])>0:
            #     classe =
            # else:
            #     classe = 0
                # continue

            (x1,y1,x2,y2) = self.convert_polygon_to_rectangle(
            [(int(obj["x1"]),int(obj["y1"])),
            (int(obj["x2"]),int(obj["y2"])),
            (int(obj["x3"]),int(obj["y3"])),
            (int(obj["x4"]),int(obj["y4"]))])

            new_x1 = max(x1, crop[1])-crop[1]
            new_y1 = max(y1, crop[0])-crop[0]
            new_x2 = min(x2, crop[1]+crop[2])-crop[1]
            new_y2 = min(y2, crop[0]+crop[3])-crop[0]
            if new_x1 > new_x2 or new_y1 > new_y2:
                # intersection empty
                pass
            elif self.filter:
                if new_x1==0 or new_y1==0 or new_x2==crop[2] or new_y2==crop[3]:
                    pass
                else:
                    return True
                    # objs.append((id_obj,(new_x1,new_y1,new_x2,new_y2),classe))
            else:
                return True
                # objs.append((id_obj,(new_x1,new_y1,new_x2,new_y2),classe))
        return False
        # return objs

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and osp.isdir(path):
            pass
        else:
            raise

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    import re
    value = unicodedata.normalize('NFKD', value).encode('utf-8', 'ignore')
    value = str(re.sub('[^\w\s-]', '', value.decode("utf-8") ).strip().lower())
    value = str(re.sub('[-\s]+', '-', value))
    return value

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def bak_load_bbox_from_csv(bbox_csv_paths):
    bbox_dict = {}
    for bbox_path in bbox_csv_paths:
        bbox_csvfile = open(bbox_path, 'r')
        # with open(pathToGT, 'r') as csvfile:
        bbox_reader = csv.DictReader(bbox_csvfile)#, delimiter=',')
        for row in bbox_reader:
            bbox_dict.setdefault(row['filename'],[]).append(row)
    return bbox_dict

def load_bbox_from_csv(bbox_csv_paths):
    bbox_dict = {}
    for bbox_path in bbox_csv_paths:
        bbox_csvfile = open(bbox_path, 'r')
        # with open(pathToGT, 'r') as csvfile:
        bbox_reader = csv.DictReader(bbox_csvfile)#, delimiter=',')
        for bbox in bbox_reader:
            filename = bbox['filename']
            label = 0
            x1,y1,x2,y2 = convert_polygon_to_rectangle(((bbox['x1'],bbox['y1']),(bbox['x2'],bbox['y2']),(bbox['x3'],bbox['y3']),(bbox['x4'],bbox['y4'])))

            if filename not in bbox_dict:
                bbox_dict[filename]={}
            if label not in bbox_dict[filename]:
                bbox_dict[filename][label]=[]

            if 'score' in bbox.keys():
                bbox_dict[filename][label].append((x1,y1,x2,y2,bbox['score']))
            else:
                bbox_dict[filename][label].append((x1,y1,x2,y2))
    return bbox_dict

def convert_polygon_to_rectangle(points):
    """points = [(x,y),(x,y),(x,y)...]"""
    x1=points[0][0]
    y1=points[0][1]
    x2=points[0][0]
    y2=points[0][1]
    for point in points:
        if point[0]<x1:
            x1=point[0]
        if point[0]>x2:
            x2=point[0]
        if point[1]<y1:
            y1=point[1]
        if point[1]>y2:
            y2=point[1]
    return (x1,y1,x2,y2)

def evaluate_csv(path_csv_detection, path_csv_groundtruth, iou_threshold=0.5, score_threshold=0.05, debug = False):
    """ Evaluate a a csv file based on another csv file.
    """
    # num_classes = 1
    # gather all detections and annotations
    # bbox_detections     = load_bbox_from_csv(path_csv_detection)
    # bbox_annotations    = load_bbox_from_csv(path_csv_groundtruth)



    # filenames = bbox_annotations.keys() #set([bbox['filename'] for bbox in bbox_annotations])

    #
    #
    # all_annotations = {}
    # for bbox in bbox_annotations:
    #     if 'status' in bbox:
    #         if not bbox['status']=='done':
    #             continue
    #     if bbox['filename'] not in all_annotations:
    #         all_annotations[bbox['filename']]={}
    #     label = 0 #to infere from source if not only one label !
    #     if label not in all_annotations[bbox['filename']]:
    #         all_annotations[bbox['filename']][label]=[]
    #
    #     x1,y1,x2,y2 = convert_polygon_to_rectangle((bbox['x1'],bbox['y1']),(bbox['x2'],bbox['y2']),(bbox['x3'],bbox['y3']),(bbox['x4'],bbox['y4']))
    #     all_annotations[bbox['filename']][label].append((x1,y1,x2,y2))
    #
    # all_detections = {}
    # for filename in bbox_detections:
    #     if filename not in all_detections:
    #         all_detections[filename]={}
    #         # all_detections.setdefault(filename,[])
    #     label = 0 #to infere from source if not only one label !
    #     if label not in all_detections[bbox['filename']]:
    #         all_detections[bbox['filename']][label]=[]
    #
    #     x1,y1,x2,y2 = convert_polygon_to_rectangle((bbox['x1'],bbox['y1']),(bbox['x2'],bbox['y2']),(bbox['x3'],bbox['y3']),(bbox['x4'],bbox['y4']))
    #     score = bbox['score']
    #     all_detections[bbox['filename']][label].append((x1,y1,x2,y2,score))

    all_detections = load_bbox_from_csv(path_csv_detection)
    all_annotations = load_bbox_from_csv(path_csv_groundtruth)
    num_classes = 1
    filenames = all_annotations.keys()

    average_precisions = {}
    results = {}
    # process detections and annotations
    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for filename in filenames:
            if filename not in all_detections:
                detections = np.array([])
            else:
                detections           = np.array(all_detections[filename][label],dtype=np.double)
            annotations          = np.array(all_annotations[filename][label],dtype=np.double)
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        jg_fp = np.sum(false_positives)
        jg_tp = np.sum(true_positives)

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)



        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        jg_recall = jg_tp / num_annotations
        jg_precision = jg_tp / np.maximum(jg_tp + jg_fp, np.finfo(np.float64).eps)
        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        results[label] = average_precision, jg_precision, jg_recall, num_annotations, jg_tp+jg_fp, jg_tp, jg_fp,
        if debug:
            print("ap",average_precision," p:",jg_precision," r:",jg_recall," numD:",jg_tp+jg_fp)
    return results

def make_img_infos(saving_folder = "",
                    folders = [],
                    groundtruth_paths= [],
                    overlap = 0.5,
                    dim = 1000,
                    scales = [1., 1.25, 0.8, 0.6, 0.4],
                    force_new_dataset = False,
                    filter = True,
                    full_image = False,
                    prediction_paths=[],
                    fileExtensions = ['ppm']):

    slug_file_name = slugify(str(folders)+'_'+str(groundtruth_paths)+'_'+str(overlap)+'_'+str(dim)+'_'+str(scales)+'_'+str(filter)+'_'+str(full_image)+'_'+str(prediction_paths)+'_'+str(fileExtensions))+".p"

    saving_path = osp.join(saving_folder,slug_file_name) #'dataset_dict.p')
    mkdir_p(saving_folder)

    if osp.isfile(saving_path) and not force_new_dataset:

        with open(saving_path, 'rb') as handle:
            print("Loading from precedent scan ["+saving_path+"]")
            img_infos = pickle.load(handle)
    else:
        img_infos={}

        img_infos['masks']=[]
        img_infos['masks_pred']=[]
        img_infos['filenames']=[]
        img_infos['size']=[]
        img_infos['spk']=[]
        img_infos['groundtruth']=[]
        img_infos['list_crops']=[]
        img_infos['prediction']=[]

        # 1 scan folders for images
        # fileExtensions = ['jpg','jpeg','JPG','JPEG']

        listOfFiles = []
        for folder in folders:
            for extension in fileExtensions:
                listOfFiles.extend( glob.glob( folder+"/**/*." + extension, recursive=True ))

        img_infos['paths']=listOfFiles

        # 2.1 load groundtruth
        groundtruth_dict = {}
        for groundtruth_path in groundtruth_paths:
            groundtruth_csvfile = open(groundtruth_path, 'r')
            # with open(pathToGT, 'r') as csvfile:
            groundtruth_reader = csv.DictReader(groundtruth_csvfile)#, delimiter=',')

            for row in groundtruth_reader:
                groundtruth_dict.setdefault(row['filename'],[]).append(row)

        # 2.2 load prediction
        prediction_dict = {}
        for prediction_path in prediction_paths:
            prediction_csvfile = open(prediction_path, 'r')
            # with open(pathToGT, 'r') as csvfile:
            prediction_reader = csv.DictReader(prediction_csvfile)#, delimiter=',')

            for row in prediction_reader:
                prediction_dict.setdefault(row['filename'],[]).append(row)

        # 3 process each image for crop generation
        scales.sort()
        for img_id,img_path in enumerate(tqdm(img_infos['paths'],desc='Img and gt loading',ncols=80)):

            filename = osp.splitext(osp.splitext(osp.basename(img_path))[0])[0]
            #filename = osp.splitext(osp.basename(img_path))[0]

            img_pil = Image.open(img_path)
            img_infos['filenames'].append(filename)
            img_infos['size'].append(img_pil.size)
            if filename not in groundtruth_dict:
                gt_dict = []
            else:
                gt_dict = groundtruth_dict[filename]
            img_infos['groundtruth'].append(gt_dict)
            if filename not in prediction_dict:
                pred_dict = []
            else:
                pred_dict = prediction_dict[filename]
            img_infos['prediction'].append(pred_dict)

            validations = []
            objs = [obj for obj in img_infos['groundtruth'][-1] if obj['filename'] == filename and obj['status']=='done']
            img_infos['masks'].append(mask(filename,objs,validations,filter=filter))


            validations_pred = []
            objs_pred = [obj for obj in img_infos['prediction'][-1] if obj['filename'] == filename and obj['status']=='done']
            img_infos['masks_pred'].append(mask(filename,objs_pred,validations_pred,filter=filter))



            # calcul du meilleur stride (sp) et du nombre de crops possibles (k) dans chaque direction (x,y)
            W = img_pil.size[0]
            H = img_pil.size[1]

            if full_image:
                scale = 1
                xywh = (0, 0, W,H)
                has_vt = img_infos['masks'][-1].contains_obj(xywh)
                crop = (img_id,scale,xywh,has_vt)
                img_infos['list_crops'].append(crop)

            else:
                spk_img=[]
                for scale in scales:
                    spk_img.append((scale,(make_spk(W, dim, overlap, scale),make_spk(H, dim, overlap, scale))))#def make_spk(W, w, overlap, scale):
                img_infos['spk'].append(spk_img)

                # crops creation
                for spk in spk_img:
                    scale = spk[0]
                    spy = spk[1][0][0] # stride horizontal
                    spx = spk[1][1][0] # stride vertical
                    for kyi in range(0,spk[1][0][1]+1):
                        yi=spy*kyi
                        for kxi in range(0,spk[1][1][1]+1):
                            xi=spx*kxi
                            xywh = (int(xi),int(yi),ceil(dim*scale),ceil(dim*scale))
                            has_vt = img_infos['masks'][-1].contains_obj(xywh)
                            crop = (img_id,scale,xywh,has_vt)
                            img_infos['list_crops'].append(crop)


        img_infos['crops_with_objs'] = [id for id,crop in enumerate(img_infos['list_crops']) if crop[3]==1]
        img_infos['crops_without_objs'] = [id for id,crop in enumerate(img_infos['list_crops']) if crop[3]==0]
        img_infos['crops_all'] = [id for id,crop in enumerate(img_infos['list_crops'])]

        print("Saving this scan")
        with open(saving_path, 'wb') as handle:
            pickle.dump(img_infos, handle, protocol = pickle.HIGHEST_PROTOCOL)

    print("                Nb img :",len(img_infos['paths']))
    print("              Nb crops :",len(img_infos['list_crops']),"["+str(int(float(len(img_infos['list_crops']))/float(len(img_infos['paths']))))+" crops/img]")
    print("   Nb crops with obj :",len(img_infos['crops_with_objs']))
    print("Nb crops without obj :",len(img_infos['crops_without_objs']))
    #pickle.dump( img_infos, open( saving_path, "wb" ) )
    return img_infos

class ALANGenerator(Generator):
    """ Generate data for a ALAN dataset.
    """

    def __init__(self, split = "train", source = "all", eval=False, overlap = 0.5, force_new_dataset = False,**kwargs):

        # self.batch_size = int(batch_size)

        ppm_folder = "/dds/work/workspace/alan_ppm_files/"
        gt_folder = "/dds/work/workspace/alan_gt_files/"

        saving_folder = "/dds/work/workspace/alan_tmp_files/"


        if source == "all":
            groundtruth_paths = [   gt_folder+ "VT_DAM3BR.csv",
                                    gt_folder+ "VT_CAT1BR.csv",
                                    gt_folder+ "SAL1BR_6021_VT.CSV"]
        elif source == "sal1":
            groundtruth_paths = [   gt_folder+ "SAL1BR_6021_VT.CSV"]
        elif source == "cat1dam3":
            groundtruth_paths = [   gt_folder+ "VT_DAM3BR.csv",
                                    gt_folder+ "VT_CAT1BR.csv",]
        elif source == "cat1":
            groundtruth_paths = [   gt_folder+ "VT_CAT1BR.csv",]
        elif source == "dam3":
            groundtruth_paths = [   gt_folder+ "VT_DAM3BR.csv",]



        folders = [ ppm_folder ]

        self.dim = 1000
        # scales = [0.8, 1, 1.2]
        scales = [1]
        # seed_train = 21


        if eval:
            filter=False
        else:
            filter = True

        full_image = False
        prediction_paths = []

        self.img_infos = make_img_infos(saving_folder = saving_folder,
                        folders = folders,
                        groundtruth_paths = groundtruth_paths,
                        overlap = overlap,
                        dim = self.dim,
                        scales = scales,
                        force_new_dataset = force_new_dataset,
                        filter = filter,
                        full_image = full_image,
                        prediction_paths=prediction_paths)

        self.classes = {"etiquette":0}
        self.labels = {0:"etiquette"}

        if eval:
            self.data = self.img_infos['crops_all']
            print("Using all crops for evaluation [",len(self.data),"].")
        else:

            train,val = self.produce_splits()
            if split=="train":
                self.data = train
            elif split=="val":
                self.data = val
            elif split=="full":
                self.data = train+val
            else:
                print("Error, no split selected [train or val or full (=train+val)]")
                raise
            print("Using",split,"split [",len(self.data),"]")

        super(ALANGenerator, self).__init__(**kwargs)

    def produce_splits(self,ratio_train_val=5, ratio_full_empty=5):
        all_crops = self.img_infos['crops_with_objs'].copy()
        # random.seed(21)
        random.shuffle(all_crops)
        train_full = all_crops[:(ratio_train_val-1)*len(all_crops)//ratio_train_val]
        train_empty = random.sample(self.img_infos['crops_without_objs'],len(train_full)//ratio_full_empty)
        train = train_full + train_empty
        random.shuffle(train)
        print("In train [",len(train),"], crops with objs [",len(train_full),"], crops without objs [",len(train_empty),"]")
        val_full = all_crops[(ratio_train_val-1)*len(all_crops)//ratio_train_val:]
        val_empty = random.sample(self.img_infos['crops_without_objs'],len(val_full)//ratio_full_empty)
        val = val_full + val_empty
        random.shuffle(val)
        print("In val [",len(val),"], crops with objs [",len(val_full),"], crops without objs [",len(val_empty),"]")
        return train,val

    # def produce_mix(self, ratio=3):
    #     """ Function to call in order to produce a new mix between list of crop with objs and list of crop without obj"""
    #     print("Producing a mix dataset of crops with objs [",len(self.img_infos['crops_with_objs']),"] and empty crops [",len(self.img_infos['crops_with_objs'])//ratio,"]")
    #     return self.img_infos['crops_with_objs'] + random.sample(self.img_infos['crops_without_objs'],len(self.img_infos['crops_with_objs'])//ratio)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.data)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes.keys())

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, sample_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        img_id, scale, xywh, has_vt = self.img_infos['list_crops'][self.data[sample_index]] #crop = (img_id,scale,xywh,has_vt)
        return float(xywh[2]) / float(xywh[3])

    def load_image(self, sample_index):
        """ Load an image at the sample_index.
        """
        img_id, scale, xywh, has_vt = self.img_infos['list_crops'][self.data[sample_index]] #crop = (img_id,scale,xywh,has_vt)
        img_path = self.img_infos['paths'][img_id]
        try:
            return np.array(pil_resize(load_crop(xywh,img_path),self.dim))
        except:
            print("Error in sample_index", img_path, img_id, scale, xywh, has_vt)

            raise

    def load_annotations(self, sample_index):
        """ Load annotations for a sample_index.
        """
        img_id, scale, xywh, has_vt = self.img_infos['list_crops'][self.data[sample_index]] #crop = (img_id,scale,xywh,has_vt)
        objs = self.img_infos['masks'][img_id].get_objs_in_crop(xywh)
        boxes  = np.zeros((len(objs), 5))
        for idx,(obj_id,obj_intersection,classe,score) in enumerate(objs):
            boxes[idx, 0] = (1./scale)*float(obj_intersection[0])
            boxes[idx, 1] = (1./scale)*float(obj_intersection[1])
            boxes[idx, 2] = (1./scale)*float(obj_intersection[2])
            boxes[idx, 3] = (1./scale)*float(obj_intersection[3])
            boxes[idx, 4] = classe
        return boxes

class TestSequence(Sequence):
    """  A sequence to test a UNIQUE image    """

    def __init__(self, image_path = "", folder_crops = "",**kwargs):

        assert len(image_path)>0
        assert len(folder_crops)>0

        self.dim = 1000
        self.image_path = image_path
        self.filename = ntpath.basename(image_path).split('.')[0]
        self.folder_crops = folder_crops
        mkdir_p(self.folder_crops)

        self.crops = self.compute_crops()
        self.produce_crops()
        # self.existing_crop_files = self.scan(folder_crops) # list(xywh)
        self.results = [None for  _ in self.crops]

    def scan(self, folder=None, ext=["jpg"]):
        if folder is None:
            folder=self.folder_crops
        listOfFiles = []
        for extension in ext:
            listOfFiles.extend( glob.glob( folder+"/**/*." + extension, recursive=True ))
        print("Found",len(listOfFiles),"files in",folder+".")
        return listOfFiles

    def path_crop(self, xywh, duplicate=False):
        # 1) get filename from file_path
        # 2) concat folder_crops/filename_xywh.jpg

        if duplicate:
            path = os.path.join(self.folder_crops, self.filename+"_"+slugify(str(xywh))+"_"+str(time.time())+".jpg")
        else:
            path = os.path.join(self.folder_crops, self.filename+"_"+slugify(str(xywh))+".jpg")
        return path


    def compute_crops(self):
        """ Compute the different possibility of xywh based on image size, sample dim and scales"""
        scales = [1]
        dim = self.dim
        overlap = 0.1
        img_pil = Image.open(self.image_path)

        W = img_pil.size[0]
        H = img_pil.size[1]
        spk_img=[]
        for scale in scales:
            spk_img.append((scale,(make_spk(W, dim, overlap, scale),make_spk(H, dim, overlap, scale))))#def make_spk(W, w, overlap, scale):


        xywh_list = []
        # crops creation
        for spk in spk_img:
            scale = spk[0]
            spy = spk[1][0][0] # stride horizontal
            spx = spk[1][1][0] # stride vertical
            for kyi in range(0,spk[1][0][1]+1):
                yi=spy*kyi
                for kxi in range(0,spk[1][1][1]+1):
                    xi=spx*kxi
                    xywh = (int(xi),int(yi),ceil(dim*scale),ceil(dim*scale))
                    xywh_list.append(xywh)
        return xywh_list

    def produce_crops(self, xywh=None, duplicate=False):
        if xywh is not None:
            path = path_crop(xywh, duplicate=duplicate)
            command="convert '"+self.image_path+"' -crop "+str(xywh[2])+"x"+str(xywh[3])+"+"+str(xywh[1])+"+"+str(xywh[0])+" '"+path+"'"
            # print("Command :",command)
            os.system(command)
            print("done.")
            #x_sizexy_size+x_offset+y_offset
            #xywh (ligne colonne w h) xywh[2]
            return path
        else:
            command = "convert '"+self.image_path+"'"
            for xywh in self.crops:
                path = self.path_crop(xywh)
                if not os.path.isfile(path):
                    command+=" \( -clone 0 -crop "+str(xywh[2])+"x"+str(xywh[3])+"+"+str(xywh[1])+"+"+str(xywh[0])+" +write '"+path+"' +delete \)"
            command+=" null:"
            if str(self.dim) not in command:
                pass
                # print("Command useless:",command)
            else:
                # print("Command :",command)
                print("Wait...")
                os.system(command)
                print("done.")


    def __len__(self):
        return len(self.crops)

    def __getitem__(self, id):
        """ Load an image at the sample_index.
        """
        xywh = self.crops[id]
        path = self.path_crop(xywh)

        if not os.path.isfile(path):
            path = self.produce_crops(xywh, duplicate=True)

        crop = Image.open(path)
        return id, xywh, np.array(crop)

class prediction_saver:
    def __init__(self, saving_folder, filename=None):
        self.saving_folder = saving_folder
        self.filename = filename
        self.keys =['filename','index','x1','y1','x2','y2','x3','y3','x4','y4','status','equipment','extra_equipment','score']
        self.keys_tagbrowser =['filename','index','x1','y1','x2','y2','x3','y3','x4','y4']
        # exemple:        CAT_1_6002,14,2173,6612,2173,6534,2505,6534,2505,6612,done,1RCV356VN,
        self.predictions = []
        self.index = 0

    def save(self):
        if self.filename is not None:
            path_csvfile = os.path.join(self.saving_folder,"results_"+self.filename+"_"+time.strftime("%Y_%m_%d-%Hh%Mmn%Ss")+"_nonms.csv")
        else:
            path_csvfile = os.path.join(self.saving_folder,"results_"+time.strftime("%Y_%m_%d-%Hh%Mmn%Ss")+"_nonms.csv")

        return self.save_data_to_csv_files(
            path_csvfile,
            self.predictions, self.keys)

    def save_tagbrowser(self):
        if self.filename is not None:
            path_csvfile = os.path.join(self.saving_folder,"results_"+self.filename+"_"+time.strftime("%Y_%m_%d-%Hh%Mmn%Ss")+"_nonms_tagbrowser.csv")
        else:
            path_csvfile = os.path.join(self.saving_folder,"results_"+time.strftime("%Y_%m_%d-%Hh%Mmn%Ss")+"_nonms_tagbrowser.csv")
        return self.save_data_to_csv_files(
            path_csvfile,
            self.predictions,self.keys_tagbrowser)

    def save_nms(self,nms_threshold = 0.01):
        filenames = list(set([p['filename'] for p in self.predictions]))
        kept_indexes = []
        for filename in filenames:
            boxes = []
            scores = []
            source_indexes = []
            for id_p,p in enumerate(self.predictions):
                if p['filename']==filename:
                    source_indexes.append(id_p)
                    boxes.append([p['y1'],p['x1'],p['y3'],p['x3']])
                    scores.append(p['score'])
            boxes=np.asarray(boxes)
            scores=np.asarray(scores)
            kept_indexes_image = non_max_suppression_fast_score(boxes, nms_threshold, scores)
            for k in kept_indexes_image:
                kept_indexes.append(source_indexes[k])

        if self.filename is not None:
            path_csvfile = os.path.join(self.saving_folder,"results_"+self.filename+"_"+time.strftime("%Y_%m_%d-%Hh%Mmn%Ss")+"_nms"+slugify(str(nms_threshold))+".csv")
        else:
            path_csvfile = os.path.join(self.saving_folder,"results_"+time.strftime("%Y_%m_%d-%Hh%Mmn%Ss")+"_nms"+slugify(str(nms_threshold))+".csv")
        return self.save_data_to_csv_files(
            path_csvfile,
            [self.predictions[i] for i in kept_indexes],self.keys)

    def save_nms_tagbrowser(self, nms_threshold = 0.01):
        filenames = list(set([p['filename'] for p in self.predictions]))
        kept_indexes = []
        for filename in filenames:
            boxes = []
            scores = []
            source_indexes = []
            for id_p,p in enumerate(self.predictions):
                if p['filename']==filename:
                    source_indexes.append(id_p)
                    boxes.append([p['y1'],p['x1'],p['y3'],p['x3']])
                    scores.append(p['score'])
            boxes=np.asarray(boxes)
            scores=np.asarray(scores)
            kept_indexes_image = non_max_suppression_fast_score(boxes, nms_threshold, scores)
            for k in kept_indexes_image:
                kept_indexes.append(source_indexes[k])

        if self.filename is not None:
            path_csvfile = os.path.join(self.saving_folder,"results_"+self.filename+"_"+time.strftime("%Y_%m_%d-%Hh%Mmn%Ss")+"_nms"+slugify(str(nms_threshold))+"_tagbrowser.csv")
        else:
            path_csvfile = os.path.join(self.saving_folder,"results_"+time.strftime("%Y_%m_%d-%Hh%Mmn%Ss")+"_nms"+slugify(str(nms_threshold))+"_tagbrowser.csv")
        return self.save_data_to_csv_files(
            path_csvfile,
            [self.predictions[i] for i in kept_indexes],self.keys_tagbrowser)

    def save_data_to_csv_files(self, path_csvfile, dict_list, keys):
        try:
            mkdir_p(os.path.dirname(path_csvfile))
            with open(path_csvfile, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys, delimiter=',',quoting=csv.QUOTE_NONE, extrasaction='ignore')
                writer.writeheader()
                for dict in dict_list:
                    writer.writerow(dict)
            print("Saved under: "+path_csvfile)
            return path_csvfile
        except:
            print("Saving did not work.")
            return None

    def add_instances(self, filename, xywh_crop, boxes, scores=None):
        """

        """
        crop_xo, crop_yo, _, _ = xywh_crop
        for i in range(boxes.shape[0]):

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            # x1, y1, x2, y2 = boxes[i] #ref matplotlib dans crop
            y1, x1, y2, x2 = boxes[i] #ref matplotlib dans crop
            xywh = (crop_xo+x1, crop_yo+y1, y2 - y1, x2 - x1) #xywh_obj_dans_full_image
            # obj
            score = scores[i] if scores is not None else None


            equipment = 'alan'
            extra_equipment = ''

            detection_dict = {'filename':filename,
                              'index':self.index,
                              'y1':int(round(xywh[0])), 'x1':int(round(xywh[1])),
                              'y2':int(round(xywh[0])), 'x2':int(round(xywh[1]+xywh[2])),
                              'y3':int(round(xywh[0]+xywh[3])),'x3':int(round(xywh[1]+xywh[2])),
                              'y4':int(round(xywh[0]+xywh[3])),'x4':int(round(xywh[1])),
                              'status':'done',
                              'equipment':equipment,
                              'extra_equipment':extra_equipment,
                              'score':score}
            self.index+=1
            self.predictions.append(detection_dict)
