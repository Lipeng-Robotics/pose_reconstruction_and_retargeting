# Camera calibration
- Camera intrinsic matrix is:
```
[[693.91839599609375, 0.0, 665.73150634765625],
[0.0, 693.91839599609375, 376.775787353515625],
[0.0, 0.0, 1.0]]
```  
- Camera extrinsic matrix could be calculated by 
```python
python cam_calib/annotator_camextr.py --img_file ${YOUR_IMAGE_PATH} --save_folder ${YOUR_SAVE_PATH}
```

`--img_file` is the path of the image to be annotated. 
`--save_folder` is the folder path of the calculated parameters to be saved.

After annotating the ground corners `ABCD` (❗be sure to follow `ABCD` order), the camera extrinsic parameters (world to camera transformation) will be computed with Perspective-N-Point algorithm. Camera intrinsic matrix `CamIntr.txt` and extrinsic matrix `CamExtr.txt` will be saved in `${YOUR_SAVE_PATH}`. 

![cam_calib](./cam_calib.png)
  