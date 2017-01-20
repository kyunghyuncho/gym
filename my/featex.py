import numpy

import theano
from theano import tensor

from collections import OrderedDict
from utils import *


def conv(x, args):
    shp_ = x.shape
    image_shp = numpy.copy(args[0])
    inp_ = x.reshape([shp_[0]*shp_[1],image_shp[0],
                      image_shp[1],image_shp[2]]).dimshuffle(0,3,1,2)
    fs_ = 3

    params = OrderedDict()

    for ii, a_ in enumerate(args[1:]):
        rs, fs, ms, activ = a_[0], a_[1], a_[2], a_[3]

        params['cW{}'.format(ii)] = theano.shared(0.01 * numpy.random.randn(fs, fs_, rs, rs).astype('float32'))
        params['cb{}'.format(ii)] = theano.shared(numpy.zeros(fs).astype('float32'))

        inp_ = tensor.nnet.conv2d(inp_, params['cW%d'%ii])
        inp_ = eval(activ)(inp_ + params['cb{}'.format(ii)][None,:,None,None])
        image_shp[0] = image_shp[0] - rs + 1
        image_shp[1] = image_shp[1] - rs + 1
        inp_ = tensor.signal.pool.pool_2d(inp_, [ms, ms], ignore_border=True)
        image_shp[0] = int(numpy.floor(numpy.float32(image_shp[0]) / ms))
        image_shp[1] = int(numpy.floor(numpy.float32(image_shp[1]) / ms))
        image_shp[2] = fs

        fs_ = fs

    inp_ = inp_.dimshuffle(0,2,3,1)
    inp_ = inp_.reshape([shp_[0],shp_[1],-1])

    print '!!!', image_shp

    return inp_, int(numpy.prod(image_shp)), params


def nothing(x, args):
    return x, None, OrderedDict()

