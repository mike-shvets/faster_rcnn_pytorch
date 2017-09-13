import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
#import torch.cuda

from faster_rcnn.datasets.SpaceNet_utils.io_images import io

def test():
    import os
    #print(torch.cuda.current_device())
    #torch.cuda.device(2)
    #print(torch.cuda.current_device())
    
    #im_file = 'demo/004545.jpg'

    im_file = '../../../data/SpaceNet/processed_asVOC_AOI_2/images/00013183.tif'
    # im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
    # im_file = '/media/longc/Data/data/2DMOT2015/test/ETH-Crossing/img1/000100.jpg'
    #image = cv2.imread(im_file)
    image = io.get_normalized_3band(im_file)


    #model_file = 'models/VGGnet_fast_rcnn_iter_70000.h5'
    #model_file = 'models/saved_model3/faster_rcnn_100000.h5'
    model_file = 'models/saved_SpaceNet/faster_rcnn_35000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch3/faster_rcnn_100000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch2/faster_rcnn_2000.h5'
    print('builing the model...')
    detector = FasterRCNN(classes=np.array(['__background__',  # always index 0
                         'building']))
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')
    
    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')

    t = Timer()
    t.tic()
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
    dets, scores, classes = detector.detect(image, 0.7)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    #im2show = np.array(image.shape, dtype='uint8')
    image = np.round(image).astype('uint8')
    im2show = np.zeros(image.shape, 'uint8')
    im2show[:] = image
    #im2show = np.copy(image.astype('uint8'))
    #im2show = np.copy(im2show)

    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        print im2show.shape
        print im2show.dtype
        print det
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join('demo', 'out.jpg'), im2show)
    #cv2.imshow('demo', im2show)
    #cv2.waitKey(0)


if __name__ == '__main__':
    test()