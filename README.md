# Image-Resizing-Optimization
Implent [Patch-Based Image Warping for Content-Aware Retargeting](http://graphics.csie.ncku.edu.tw/Tony/papers/IEEE_Multimedia_resizing_2013_Feb.pdf) algorithm

## Code modified from
* https://github.com/tobygameac/Content-Aware-Retargeting
* https://github.com/pochih0313/Patch-Based-Image-Warping-for-Content-Aware-Retargeting?fbclid=IwAR2ubV83as3U8KTudNNVWuDLhnEQKWxgGoQocydA7aEkzY5YtPzaP_obieA

## Main modification
* Make mesh color different according to its significance value
* Remove boarder line caused by `cv::warpPerspective`

## Result
* source image:

![dog](https://user-images.githubusercontent.com/57750932/200822562-a27cd972-3963-405a-8179-74d52a307746.jpg)

* Patch-Based Image Warping:

![1280X366](https://user-images.githubusercontent.com/57750932/200822594-af8b2937-be49-49de-b923-8d231e2b7c34.png)

* Patch-Based Image Warping with mesh:

![1280X366patch](https://user-images.githubusercontent.com/57750932/200822584-6d1c8ab3-1008-4b7b-8007-a0af4304991f.png)

## Note
https://hackmd.io/k3cS6MkKRCioS0yYGj1bYw
