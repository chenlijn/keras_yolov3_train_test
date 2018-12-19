it is a repository for training optic disc detector using keras-yolo3

# environment  
use Python3, tensorflow and keras to train.  
Both Python2 and Python3 accompanied with tensorflow can deploy the frozen model.  
  

# training procedure  

1. convert the cfg and weights into h5 
2. prepare the images as txt files in folder lijian\_work
3. prepare the cfg and pretrained weights 
4. prepare the anchors and class names
5. modify train\_disc\_detection.py and train
  
  

# save the model as tf Graph  
1. in the deploy folder, prepare a test image as specified in run\_yolo\_and\_save\_graph.py
2. run the one\_step\_to\_prepare\_frozen\_model.py
3. then everything is done, use the deploy\_frozen\_model.py to detect your targets.
  
  

# how to deploy  
You only need the forzen '\*.pd' file and the script deploy\_frozen\_model.py
