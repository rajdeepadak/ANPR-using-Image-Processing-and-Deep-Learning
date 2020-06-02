import sys, os
import argparse, time
from yolo import YOLO, detect_video
from PIL import Image
import random, cv2
import numpy as np
from recognition import preprocess, apply_morphology, detect_chars

base_path = 'E:/License-Plate-Recognition-System/Test Set/'
save_dir = 'result'

def detect_img(yolo):

    for subdirs,dirs,files in os.walk(base_path):
        count=0
        for filename in files:
            start = time.time()
            try:
                image = Image.open(base_path+filename)
            except:
                print('Open Error! Try again!')
                continue
            else:
                try:
                    original_image = image.copy()
                    r_image, left, top, right, bottom = yolo.detect_image(image)
                    c_image = original_image.crop((left, top, right, bottom))

                    r_image.save(save_dir + '/' + 'Detected' + '/' + 'detected_' + str(count) + '.png')
                    # c_image.save(save_dir + '/' + 'Cropped' + '/' + 'cropped_pic' + str(count) + '.png')
                    
                    cv2.imshow('Original Image', cv2.resize(np.array(original_image),(416,416)))
                    cv2.imshow('Detected plate', cv2.resize(np.array(r_image),(416,416)))

                    cropped_plate = np.array(c_image)

                    resized_img, canny_img = preprocess(cropped_plate)

                    morph_img = apply_morphology(canny_img)
                    

                    number = detect_chars(morph_img, resized_img)
                    end = time.time()
                    print('Time taken:',end - start)
                    print('Predicted number:',number)

                    cv2.imwrite(save_dir+'/Recognized/'+'recog_' + str(count)+'_'+number+'.png', cropped_plate)

                    cv2.waitKey(0)
                    cv2.destroyAllWindows


                except:
                    print('Number Plate not detected.')
            count+=1

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
