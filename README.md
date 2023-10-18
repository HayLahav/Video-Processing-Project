# Video-Processing-Project
This project involved implementing a comprehensive video processing pipeline, including stabilization, background subtraction, matting, and object tracking,


                                                                                                                                

![image](https://github.com/HayLahav/Video-Processing-Project/assets/111200362/b60ea6c3-983e-41c4-86a2-04365e49298b)

In this project, we aimed to develop a robust video processing pipeline encompassing multiple stages. The initial phase involved preprocessing by converting RGB frames to the HSV color space, identifying the object's bounding box, and cropping frames to enhance the efficiency of subsequent computationally expensive functions. Following this, creating scribbles involved generating foreground and background masks based on binary frames, further refined using morphological operations.

The next key steps included creating probability maps for the object and background using Gaussian Kernel Density Estimation (KDE), calculating geodesic distance maps, and generating a narrow band and alpha trimap. The opacity map was computed by assigning weights to pixels in the narrow band based on KDE probabilities and geodesic distances. Finally, the object was seamlessly composited onto a new background using the calculated opacity map.

The tracking phase utilized two approaches: a binary mask for precise boundary tracking and a particle filter for handling body part occlusions. These methods were employed interchangeably based on the binary parameter. Overall, the project optimized runtime efficiency, resulting in a well-rounded video processing system with stable stabilization, accurate object segmentation, and reliable tracking.
