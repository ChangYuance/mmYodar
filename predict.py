
CUDA_VISIBLE_DEVICES=1
import os

import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from tqdm import tqdm

if __name__ == "__main__":
    yolo = YOLO()

    mode = "dir_predict"       
    crop            = False
    count           = False

    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    test_interval   = 100
    fps_image_path  = "img/street.jpg"

    dir_origin_path = r"test\corridor\expnew"
    dir_pct_path    = r"test\corridor\expnew"
    dir_save_path   = "predict\pct/"

    heatmap_save_path = "model_data/heatmap_vision.png"

    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            pct = input('Input pct filename:')
            try:
                image = Image.open(img)
                pct   = Image.open(pct)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
                c = cv2.waitKey(5) & 0xff
                r_image.save("img.jpg")

    elif mode == "video":   
        while True:
            video_path_file = input("Input vidoe filename:")
            try:
                capture = cv2.VideoCapture(video_path_file)
            except:
                print('Open Error! Try again!')
                continue
            else:
      
                if video_save_path != "":
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

                ref, frame = capture.read()
                if not ref:
                    raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

                fps = 0.0
                while (True):
                    t1 = time.time()
                    
                    ref, frame = capture.read()
                    if not ref:
                        break
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                   
                    frame = Image.fromarray(np.uint8(frame))
                   
                    frame = np.array(yolo.detect_image(frame))
                   
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    fps = (fps + (1. / (time.time() - t1))) / 2
                    print("fps= %.2f" % (fps))
                    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("video", frame)
                    c = cv2.waitKey(1) & 0xff
                    if video_save_path != "":
                        out.write(frame)

                    if c == 27:
                        capture.release()
                        break

                print("Video Detection Done!")
                capture.release()
                if video_save_path != "":
                    print("Save processed video to the path :" + video_save_path)
                    out.release()
                cv2.destroyAllWindows()

    elif mode == "video":    
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    
        fps = 0.0
        while(True):
            t1 = time.time()

            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(np.uint8(frame))

            frame = np.array(yolo.detect_image(frame))

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)
    
            if c==27:
                capture.release()
                break
    
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                pct_path    = os.path.join(dir_pct_path,img_name)
                image       = Image.open(image_path)
                
                pct         = Image.open(pct_path)
                # pct = np.array(pct)
                # cv2.imshow('input_image', pct)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                r_image     = yolo.detect_image(image,pct)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                print(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")))
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
