from tensorflow.core.framework import graph_pb2
import tensorflow as tf
from PIL import Image

import numpy as np
import cv2


class DiscDetection(object):

    def __init__(self):
        self._binary_model_file = 'frozen_yolov3.pb'
        self._model_input_size = (608, 608)
        self._graph_def = graph_pb2.GraphDef()
        self._box_enlarge_ratio = 0.0

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(1.0*w/iw, 1.0*h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    def regularize_cor_cut(self, ori_image, xmin, ymin, xmax, ymax):
        img_height = ori_image.shape[0]
        img_width = ori_image.shape[1]
        # Mask sure all coordinates in right area
    
        if xmin > img_width:
            xmin = img_width
        elif xmin < 0:
            xmin = 0
    
        if xmax > img_width:
            xmax = img_width
        elif xmax < 0:
            xmax = 0
    
        if ymin > img_height:
            ymin = img_height
        elif ymin < 0:
            ymin = 0
    
        if ymax > img_height:
            ymax = img_height
        elif ymax < 0:
            ymax = 0
    
        return xmin, ymin, xmax, ymax


    def ave_edge_cal(self, ori_image):
        gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
        ul = gray[:5, :5]
        ur = gray[:5, ori_image.shape[1] - 5:]
        ll = gray[ori_image.shape[0] - 5:, :5]
        lr = gray[ori_image.shape[0] - 5:, ori_image.shape[1] - 5:]
        mean = (np.sum(ul) / 25 + np.sum(ur) / 25 + np.sum(ll) / 25 + np.sum(lr) / 25) / 4
        return mean
    

    def remove_black_edge(self, ori_image):
        thresh = 15
        b, g, r = cv2.split(ori_image)
        img_width = ori_image.shape[1]
        img_height = ori_image.shape[0]
   
        mean_pixel = self.ave_edge_cal(ori_image)
        if mean_pixel < 127:
            img_mask = np.array(((b < mean_pixel+thresh) *
                                 (g < mean_pixel+thresh) *
                                 (r < mean_pixel+thresh)) * 1, dtype=np.uint8)
        else:
            img_mask = np.array(((b > mean_pixel-thresh) *
                                 (g > mean_pixel-thresh) *
                                 (r > mean_pixel-thresh)) * 1, dtype=np.uint8)
   
        halfv_h_start = np.sum(img_mask[:, :img_width / 2], 1).reshape(img_height, 1)
        half_h_end = img_width - np.sum(img_mask[:, img_width / 2:], 1).reshape(img_height, 1)
        half_h_len = max(img_width / 2 - np.min(halfv_h_start), np.max(half_h_end) - img_width / 2)
   
        half_v_start = np.sum(img_mask[:img_height / 2, :], 0).reshape(img_width, 1)
        half_v_end = img_height - np.sum(img_mask[img_height / 2:, :], 0).reshape(img_width, 1)
        half_v_len = max(img_height / 2 - np.min(half_v_start), np.max(half_v_end) - img_height / 2)
   
        xmin, ymin, xmax, ymax = self.regularize_cor_cut(ori_image,
                                                    int(img_width / 2 - half_h_len),
                                                    int(img_height / 2 - half_v_len),
                                                    int(img_width / 2 + half_h_len),
                                                    int(img_height / 2 + half_v_len))
        #cut_image = ori_image[ymin: ymax, xmin: xmax]
   
        #return cut_image
        return np.array([xmin, ymin, xmax, ymax])



    def load_network(self):
        with open(self._binary_model_file, 'rb') as f:
            self._graph_def.ParseFromString(f.read()) 
        tf.import_graph_def(self._graph_def, name ='')
        self._sess = tf.Session()



    def predict(self, image):
        """
        predict bounding boxes

        args:
            image - should be PIL image
        """

        image_size = [image.size[1], image.size[0]]
        boxed_image = self.letterbox_image(image, self._model_input_size)
        
        image_data = np.array(boxed_image, dtype='float32') 
        image_data /= 255.
        
        image_data = np.expand_dims(image_data, 0)
        
        # run model 
        boxes_, scores_, classes_= self._sess.run(['boxes:0', 'scores:0', 'classes:0'], feed_dict={'input_1:0':image_data, 'input_image_shape:0':image_size, 'batch_normalization_1/keras_learning_phase:0':0})
        
        # get the box with the largest score
        #boxes_ = boxes_.astype(int)
        if len(scores_) > 0:
            index = np.argmax(scores_)
            box = boxes_[index]
            
            # enlarge the bounding box
            top, left, bottom, right = box
            delta_w = (right - left) * self._box_enlarge_ratio 
            delta_h = (bottom - top) * self._box_enlarge_ratio 
            top = max(0, top-delta_h)
            bottom = min(image.size[1], bottom+delta_h)
            left = max(0, left-delta_w)
            right = min(image.size[0], right+delta_w)
            new_box = np.array([top, left, bottom, right])
            new_box = new_box.astype(int)
            return new_box
        else:
            return None


    def get_disc_box(self, image_file):

        image = cv2.imread(image_file)
        image_box = self.remove_black_edge(image)
        xmin, ymin, xmax, ymax = image_box
        image_remove = image[ymin:ymax, xmin:xmax, :]
        self.image_size = (image_remove.shape[0], image_remove.shape[1])

        # convert cv2 to PIL
        pil_image = Image.fromarray(image_remove[:,:,::-1])
        box = self.predict(pil_image)
        return box



if __name__=='__main__':
    
    # set up the network
    detector = DiscDetection()
    detector.load_network()

    # deploy
    #image_file = '000011.jpg'
    image_file = 'test.jpg'
    image_show = cv2.imread(image_file)
    image_box = detector.remove_black_edge(image_show)
    xmin, ymin, xmax, ymax = image_box
    image_remove = image_show[ymin:ymax, xmin:xmax, :]
    #cv2.imwrite('no_black_edge.png', image_remove)

    # convert PIL to cv2
    pil_image = Image.fromarray(image_remove[:,:,::-1])

    box = detector.predict(pil_image)
    if box is not None:
        top, left, bottom, right = box
        cv2.rectangle(image_remove, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.imwrite('det2.png', image_remove)



