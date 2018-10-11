#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import h5py

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
#        caffe.set_device(gpu_id)
#        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

        print( self.net.params['conv1_1'][0].data )

        file = h5py.File( '/Users/julian/CZModels/SSD300VGGReference20180920.hdf', 'w')

        file.create_dataset("/conv1_1W", data = self.net.params['conv1_1'][0].data )
        file.create_dataset("/conv1_2W", data = self.net.params['conv1_2'][0].data )
        file.create_dataset("/conv1_1B", data = self.net.params['conv1_1'][1].data )
        file.create_dataset("/conv1_2B", data = self.net.params['conv1_2'][1].data )

        file.create_dataset("/conv2_1W", data = self.net.params['conv2_1'][0].data )
        file.create_dataset("/conv2_2W", data = self.net.params['conv2_2'][0].data )
        file.create_dataset("/conv2_1B", data = self.net.params['conv2_1'][1].data )
        file.create_dataset("/conv2_2B", data = self.net.params['conv2_2'][1].data )

        file.create_dataset("/conv3_1W", data = self.net.params['conv3_1'][0].data )
        file.create_dataset("/conv3_2W", data = self.net.params['conv3_2'][0].data )
        file.create_dataset("/conv3_3W", data = self.net.params['conv3_3'][0].data )
        file.create_dataset("/conv3_1B", data = self.net.params['conv3_1'][1].data )
        file.create_dataset("/conv3_2B", data = self.net.params['conv3_2'][1].data )
        file.create_dataset("/conv3_3B", data = self.net.params['conv3_3'][1].data )

        file.create_dataset("/conv4_1W", data = self.net.params['conv4_1'][0].data )
        file.create_dataset("/conv4_2W", data = self.net.params['conv4_2'][0].data )
        file.create_dataset("/conv4_3W", data = self.net.params['conv4_3'][0].data )
        file.create_dataset("/conv4_1B", data = self.net.params['conv4_1'][1].data )
        file.create_dataset("/conv4_2B", data = self.net.params['conv4_2'][1].data )
        file.create_dataset("/conv4_3B", data = self.net.params['conv4_3'][1].data )

        file.create_dataset("/conv5_1W", data = self.net.params['conv5_1'][0].data )
        file.create_dataset("/conv5_2W", data = self.net.params['conv5_2'][0].data )
        file.create_dataset("/conv5_3W", data = self.net.params['conv5_3'][0].data )
        file.create_dataset("/conv5_1B", data = self.net.params['conv5_1'][1].data )
        file.create_dataset("/conv5_2B", data = self.net.params['conv5_2'][1].data )
        file.create_dataset("/conv5_3B", data = self.net.params['conv5_3'][1].data )

        file.create_dataset("/conv6_W", data = self.net.params['fc6'][0].data )
        file.create_dataset("/conv6_B", data = self.net.params['fc6'][1].data )

        file.create_dataset("/conv7_W", data = self.net.params['fc7'][0].data )
        file.create_dataset("/conv7_B", data = self.net.params['fc7'][1].data )

        file.create_dataset("/conv8_1W", data = self.net.params['conv6_1'][0].data )
        file.create_dataset("/conv8_2W", data = self.net.params['conv6_2'][0].data )
        file.create_dataset("/conv8_1B", data = self.net.params['conv6_1'][1].data )
        file.create_dataset("/conv8_2B", data = self.net.params['conv6_2'][1].data )

        file.create_dataset("/conv9_1W", data = self.net.params['conv7_1'][0].data )
        file.create_dataset("/conv9_2W", data = self.net.params['conv7_2'][0].data )
        file.create_dataset("/conv9_1B", data = self.net.params['conv7_1'][1].data )
        file.create_dataset("/conv9_2B", data = self.net.params['conv7_2'][1].data )

        file.create_dataset("/conv10_1W", data = self.net.params['conv8_1'][0].data )
        file.create_dataset("/conv10_2W", data = self.net.params['conv8_2'][0].data )
        file.create_dataset("/conv10_1B", data = self.net.params['conv8_1'][1].data )
        file.create_dataset("/conv10_2B", data = self.net.params['conv8_2'][1].data )

        file.create_dataset("/conv11_1W", data = self.net.params['conv9_1'][0].data )
        file.create_dataset("/conv11_2W", data = self.net.params['conv9_2'][0].data )
        file.create_dataset("/conv11_1B", data = self.net.params['conv9_1'][1].data )
        file.create_dataset("/conv11_2B", data = self.net.params['conv9_2'][1].data )

        file.create_dataset("/block4_classes_W", data = self.net.params['conv4_3_norm_mbox_conf'][0].data )
        file.create_dataset("/block4_classes_B", data = self.net.params['conv4_3_norm_mbox_conf'][1].data )
        file.create_dataset("/block4_loc_W", data = self.net.params['conv4_3_norm_mbox_loc'][0].data )
        file.create_dataset("/block4_loc_B", data = self.net.params['conv4_3_norm_mbox_loc'][1].data )

        file.create_dataset("/block7_classes_W", data = self.net.params['fc7_mbox_conf'][0].data )
        file.create_dataset("/block7_classes_B", data = self.net.params['fc7_mbox_conf'][1].data )
        file.create_dataset("/block7_loc_W", data = self.net.params['fc7_mbox_loc'][0].data )
        file.create_dataset("/block7_loc_B", data = self.net.params['fc7_mbox_loc'][1].data )

        file.create_dataset("/block8_classes_W", data = self.net.params['conv6_2_mbox_conf'][0].data )
        file.create_dataset("/block8_classes_B", data = self.net.params['conv6_2_mbox_conf'][1].data )
        file.create_dataset("/block8_loc_W", data = self.net.params['conv6_2_mbox_loc'][0].data )
        file.create_dataset("/block8_loc_B", data = self.net.params['conv6_2_mbox_loc'][1].data )

        file.create_dataset("/block9_classes_W", data = self.net.params['conv7_2_mbox_conf'][0].data )
        file.create_dataset("/block9_classes_B", data = self.net.params['conv7_2_mbox_conf'][1].data )
        file.create_dataset("/block9_loc_W", data = self.net.params['conv7_2_mbox_loc'][0].data )
        file.create_dataset("/block9_loc_B", data = self.net.params['conv7_2_mbox_loc'][1].data )

        file.create_dataset("/block10_classes_W", data = self.net.params['conv8_2_mbox_conf'][0].data )
        file.create_dataset("/block10_classes_B", data = self.net.params['conv8_2_mbox_conf'][1].data )
        file.create_dataset("/block10_loc_W", data = self.net.params['conv8_2_mbox_loc'][0].data )
        file.create_dataset("/block10_loc_B", data = self.net.params['conv8_2_mbox_loc'][1].data )

        file.create_dataset("/block11_classes_W", data = self.net.params['conv9_2_mbox_conf'][0].data )
        file.create_dataset("/block11_classes_B", data = self.net.params['conv9_2_mbox_conf'][1].data )
        file.create_dataset("/block11_loc_W", data = self.net.params['conv9_2_mbox_loc'][0].data )
        file.create_dataset("/block11_loc_B", data = self.net.params['conv9_2_mbox_loc'][1].data )

        file.create_dataset("/conv4_3_norm", data = self.net.params['conv4_3_norm'][0].data )

def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='data/VOC0712/labelmap_voc.prototxt')
    parser.add_argument('--model_def',
                        default='models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights',
                        default='models/VGGNet/VOC0712/SSD_300x300/'
                        'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
