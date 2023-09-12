# mmYodar
### <font color=Blue>This is the source code and dataset of the ***SECON 2023*** paper - "mmYodar: Lightweight and Robust Object Detection using mmWave Signals". <br></font>
![image](https://github.com/ChangYuance/mmYodar/blob/main/cover/cover.jpg)
****
### Overall, the mmYodar project includes four parts: <br> 1.mmWave Signal Pre-processing Process <br> 2.Real-time System <br> 3.Camera and mmWave multimodal Dataset <br> 4.Object Detection Network.
****
### 1.The mmWave signal pre-processing process of mmYodar is demonstrated in *Radar_points_image* folder.<br>
### Please `run GetFiles.m` to get radar points images which are the input of detection network. More details can be found in code.<br><br>
### Using your own mmWave binary files requires a deep understanding of mmWave data transmission and reception. Many parameters in the processing program need to be carefully modified accordingly.
### Therefore, we provide pre-collected usable mmWave files. Baidu Netdisk: <br>https://pan.baidu.com/s/13XubeLqVNEAWVo74W8ct9A?pwd=ZXCV Extracted code：ZXCV
****
### 2.The real-time system of mmYodar is demonstrated in *OnlineSystem* folder.<br>
### Please `run test.py` for visual results. After initialization, you need to press the button *DCA1000ARM* and *Trigger frame* orderly. <br>
### Noted: Online system require installing  [mmWave studio software](https://www.ti.com/tool/MMWAVE-STUDIO). Of course, you should have a mmwave radar.
****
### 3.The part of Camera and mmWave multimodal Dataset is demonstrated in *Dataset* folder. For privacy and security, please contact for a complete dataset greenthunder@stu.xjtu.edu.cn.<br>
### Please `run test.py` for visual results. After initialization, you need to press the button *DCA1000ARM* and *Trigger frame* orderly. <br>
### Therefore, we provide picture test sets of different scenes. Baidu Netdisk: <br>https://pan.baidu.com/s/13XubeLqVNEAWVo74W8ct9A?pwd=ZXCV Extracted code：ZXCV
****
### 4.Object Detection Network.<br>
### Please `run predict.py` for visual results. You need to put yolo, utils, mini_net in the same directory.After that, download the trained weights. <br>
### Therefore, we provide trained weights. <br>https://www.aliyundrive.com/s/btVjpoXBVZv Extracted code：nr35 <br>
### If you need to see the results of different test sets, please modify lines 26 and 27 in the predict.py, the data set is given in the third part.
****
## Acknowledgements
### The code for reading DCA1000 data in online system is partly borrowed from [mmmesh](https://github.com/HavocFiXer/mmMesh).
### our code borrow from https://github.com/bubbliiiing/yolo3-pytorch
****
