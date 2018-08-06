
from keras_retinanet.preprocessing.alan import evaluate_csv

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

# model 6 epoch nms 0.001
# paths_csv_detection = ["/dds/work/workspace/alan_tmp_files/multiple_files/results_multiple_files_2018_08_03-15h15mn46s_nms001.csv"]
# ap 0.5496758424830148  p: 0.2568149210903874  r: 0.7649572649572649  numD: 697.0

# model 13 epoch nms 0.001
# paths_csv_detection = ["/home/j61678/workspace/dds/work/workspace/alan_tmp_files/multiple_files/results_multiple_files_epoch13_2018_08_03-15h56mn39s_nms001.csv"]
# ap 0.576777923526475  p: 0.2896440129449838  r: 0.7649572649572649  numD: 618.0


# model 13 epoch no nms
# paths_csv_detection = ["/home/j61678/workspace/dds/work/workspace/alan_tmp_files/multiple_files/results_multiple_files_epoch13_2018_08_03-15h56mn39s_nonms.csv"]
# ap 0.4094805029238715  p: 0.21505376344086022  r: 0.7692307692307693  numD: 837.0


for iou_threshold_ in range(0,101,10):
    iou_threshold = float(iou_threshold_)/100.
    for score_threshold_ in range(0,101,10):
        score_threshold = float(score_threshold_)/100.
        print(90*" ", "iou:",iou_threshold,"| score_threshold:",score_threshold)
        evaluate_csv(paths_csv_detection, paths_csv_groundtruth, iou_threshold=iou_threshold, score_threshold=score_threshold, debug = True)

evaluate_csv(paths_csv_detection, paths_csv_groundtruth, iou_threshold=0.01, score_threshold=0.1, debug = True)
