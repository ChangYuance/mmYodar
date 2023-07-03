# mmYodar
### <font color=Blue>This is the source code and dataset of the ***SECON 2023*** paper - "mmYodar: Lightweight and Robust Object Detection using mmWave Signals". <br></font>
![image](https://github.com/ChangYuance/mmYodar/blob/main/cover/cover.jpg)
****
### Overall, the mmYodar project includes four parts: <br> 1.mmWave Signal Pre-processing Process <br> 2.Real-time System <br> 3.Camera and mmWave multimodal Dataset <br> 4.Object Detection Network.
****
### The mmWave signal pre-processing process of mmYodar is demonstrated in *Radar_points_image* folder.<br>
### Please use the main fuction GetFiles.m to get radar points images which are the input of detection network. More details can be found in code.<br><br>
### Using your own mmWave binary files requires a deep understanding of mmWave data transmission and reception.Many parameters in the processing program need to be carefully modified accordingly.
### Therefore, we provide pre-collected usable mmWave files.Baidu Netdisk:https://pan.baidu.com/s/13XubeLqVNEAWVo74W8ct9A?pwd=ZXCV Extracted code：ZXCV
****
### The real-time system of mmYodar is demonstrated in *OnlineSystem* folder.<br>
### Please run the test.py for visual results. After initialization, you need to press the button *DCA1000ARM* and *Trigger frame* orderly. <br>
### Noted: Online system require installing mmWave studio software [Ti](link:https://www.ti.com/tool/MMWAVE-STUDIO). Of course, you do need to have a mmwave radar.
****
## Acknowledgements
* ### The code of reading DCA1000 data in online system is partly borrowed from [mmmesh](https://github.com/HavocFiXer/mmMesh).
****
