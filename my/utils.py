import logging
import os, sys

import theano
from theano import tensor

import numpy
from collections import OrderedDict

def sgd(tparams, grads, inp, clip_c=1.):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()
    new_grads = []
    for g in grads:
        new_grads.append(tensor.switch(g2 > (clip_c**2),
                                       g / tensor.sqrt(g2) * clip_c,
                                       g))
    grads = new_grads

    lr0 = numpy.float32(0.0001)

    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, [], updates=gsup)

    pup = [(p, p - lr0 * g) for p, g in zip(tparams.values(), gshared)]
    f_update = theano.function([], [], updates=pup)

    return f_grad_shared, f_update

def adam(tparams, grads, inp, clip_c=1.):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()
    new_grads = []
    for g in grads:
        new_grads.append(tensor.switch(g2 > (clip_c**2),
                                       g / tensor.sqrt(g2) * clip_c,
                                       g))
    grads = new_grads

    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, [], updates=gsup, profile=False)

    lr0 = numpy.float32(0.0002)
    b1 = numpy.float32(0.1)
    b2 = numpy.float32(0.001)
    e = numpy.float32(1e-8)

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + numpy.float32(1.)
    fix1 = numpy.float32(1.) - b1**(i_t)
    fix2 = numpy.float32(1.) - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((numpy.float32(1.) - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((numpy.float32(1.) - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([], [], updates=updates,
                               on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def tanh(x):
    return tensor.tanh(x)

def relu(x):
    return tensor.maximum(0., x)

def orth(R):
    u, s, v = numpy.linalg.svd(R)
    return u.astype('float32')

def movavg(x, x0, coef=0.8):
    for nn, vv in x0.items():
        x0[nn].set_value(coef * x0[nn].get_value() + (1. - coef) * x[nn].get_value())

def make_copy(p):
    p_ = OrderedDict()
    for nn, vv in p.items():
        p_[nn] = theano.shared(vv.get_value())
    return p_

def transfer(p_from, p_to):
    for nn, vv in p_from.items():
        p_to[nn].set_value(vv.get_value())

