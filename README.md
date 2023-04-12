# VESCPN
Keras implementation of VESPCN with MSE, DCT2 , and Quantized DCT Loss functions.


This code requires keras with a tensorflow backend. 

## Usage
For training, run  
`python train_vespcn.py --scale_factor=2`
<br>
For testing, run  
`python test_vespcn.py --scale_factor=2`  
the Super-Resolution images are in the result folder

To make video from images, run
`makeVideo.py`


## References
https://github.com/anidh/espcn-keras
