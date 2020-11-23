import argparse
import logging
import time

import cv2
import numpy as np
import math

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_coords(body_part,image):
    return (int(body_part.x*image.shape[1]) , int(body_part.y*image.shape[0]))

def putText(text, image , co_ords):
    return cv2.putText(image , text , co_ords , cv2.FONT_HERSHEY_DUPLEX , 1, (255,255,255) , 2)


def get_angle(upper_coord, lower_coord):
    return math.degrees(math.atan((upper_coord[1]-lower_coord[1])/(upper_coord[0]-lower_coord[0])))

def num_in_between(num , lims):

    if num>=lims[0] and num<=lims[1]:
        return True

    else:
        return False

def diff(a , b):
    return abs(a-b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    # logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output.avi" , fourcc , 5 , (image.shape[1],image.shape[0]))


    curl_count = 0
    #prev_angle = -1000
    angles = [0, 0]
    count = 0

    left_angles = [0, 0]
    left_count = 0



    while True:
        ret_val, image = cam.read()

        # logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        for human in humans:

            #ANGLE BETWEEN NOSE AND NECK -------------------------------------------------------------------------------


            try:
                neck_angle = abs(get_angle(get_coords(human.body_parts[0], image) , get_coords(human.body_parts[1], image)))

                if num_in_between(neck_angle , (20 , 49)):
                    cv2.putText(image, "FIX NECK POSTURE" , (8,40) , cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255) , 2)

            except:
                pass
                #print("Couldnt get either neck or nose")



            #ANGLE BETWEEN LEFT-HIP AND NECK ---------------------------------------------------------------------------

            try:

                lefthip_angle = abs(get_angle(get_coords(human.body_parts[1], image) , get_coords(human.body_parts[8], image)))
                if num_in_between(lefthip_angle , (25 , 70)):
                    cv2.putText(image, "FIX BACK POSTURE" , (8,80) , cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255) ,2)

            except:
                pass
                #print("Couldnt get either lefthip or nose")


            #ANGLE BETWEEN RIGHT-HIP AND NECK --------------------------------------------------------------------------
            try:
                righthip_angle = abs(get_angle(get_coords(human.body_parts[1], image) , get_coords(human.body_parts[11], image)))
                if num_in_between(righthip_angle , (25 , 70)):
                    cv2.putText(image, "FIX BACK POSTURE" , (8,80) , cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255) ,2)

            except:
                pass
                # print("Couldnt get either righthip or nose")



            #CURL COUNTING ---------------------------------------------------------------------------

            #RIGHT HAND CURL COUNTING:

            try:
                min_angle = 0



                righthand_angle = get_angle(get_coords(human.body_parts[3], image), get_coords(human.body_parts[4], image))
                current_angle = righthand_angle
                print(f"RIGHTHAND {righthand_angle}")
                temp = angles[0]
                angles[0] = current_angle
                angles[1] = temp
                # 0th pos represents current angle and 1st pos represents previous angle

                if righthand_angle < min_angle:

                    if (angles[0] > int(0.95*angles[1])) and (diff(angles[0] , angles[1]) >8):

                        count += 1

                        if count == 1:
                            curl_count += 1
                            count = 0

                #cv2.putText(image, f"CURL COUNT = {curl_count}", (8, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)


            except:
                print("Couldnt find right elbow or wrist")


            try:
                min_angle = 20

                lefthand_angle = get_angle(get_coords(human.body_parts[6], image), get_coords(human.body_parts[7], image))
                current_angle = lefthand_angle
                print(f"LEFTHAND {lefthand_angle}")
                temp = left_angles[0]
                left_angles[0] = current_angle
                left_angles[1] = temp
                # 0th pos represents current angle and 1st pos represents previous angle

                if lefthand_angle > min_angle:
                    print("GREATER")
                    print(left_angles[0])
                    print(left_angles[1])
                    if (left_angles[0] > int(1*left_angles[1])) and (diff(left_angles[0], left_angles[1]) > 15):
                        print(f"COUNTER INCREASED")
                        left_count += 1

                        if left_count == 2:
                            curl_count += 1
                            left_count = 0

                # cv2.putText(image, f"CURL COUNT = {curl_count}", (8, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)


            except:
                print("Couldnt find elbow or wrist")




        #logger.debug('postprocess+')

        cv2.putText(image, f"CURL COUNT = {curl_count}", (8, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()

        out.write(image)

        if cv2.waitKey(10) == ord("q"):
            break
        # logger.debug('finished+')

    cv2.destroyAllWindows()
