import logging
import os, sys

import theano
from theano import tensor

import numpy
from collections import OrderedDict

from utils import *
import featex

class VApprox(object):

    def __init__(self, 
                 parent,
                 obs_dim,
                 n_hidden=100,
                 disc_factor=0.,
                 activ='tanh',
                 movavg_coeff=0.,
                 truncate_gradient=-1,
                 optimizer='adam',
                 featex=featex.nothing,
                 featex_args=None):

        self.parent = parent
        self.obs_dim = obs_dim

        self.n_hidden   = n_hidden
        self.disc_factor= disc_factor
        self.activ      = activ
        self.movavg_coeff = movavg_coeff
        self.truncate_gradient = truncate_gradient

        self.featex = featex
        self.featex_args = featex_args

        self.vars_init()

        self.param_init()
        self.forward_init()
        self.grad_init()

        self.f_shared, self.f_update = eval(optimizer)(self.vparams, 
                                                       self.cost,
                                                       self.vgrads,
                                                       [self.parent.obs, self.parent.rewards, 
                                                        self.parent.mask]) 

    def param_init(self):
        self.vparams = OrderedDict()

        self.vparams['W'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.vparams['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.vparams['U'] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, 1).astype('float32'))
        self.vparams['c'] = theano.shared(numpy.zeros(1).astype('float32'))

        for pp, vv in self.fparams.items():
            self.vparams[pp] = vv

        if self.movavg_coeff > 0.:
            self.old_vparams = make_copy(self.vparams)

    def vars_init(self):
        self.obs_, indim, self.fparams = self.featex(self.parent.obs, self.featex_args)
        self.obs_dim = indim if indim is not None else self.obs_dim

    def forward_init(self):
        obs_ = self.obs_.reshape([self.obs_.shape[0]*self.obs_.shape[1], self.obs_.shape[-1]])

        h = eval(self.activ)(tensor.dot(obs_, self.vparams['W']) + self.vparams['b'][None,:])

        self.v = tensor.dot(h, self.vparams['U']) + self.vparams['c'][None,:]
        self.vforward = theano.function([self.parent.obs], self.v, name='vforward')

    def grad_init(self):
        mask_ = self.parent.mask.flatten()
        rewards_ = self.parent.rewards.flatten()

        mean_rewards = ((mask_ * rewards_)
                        .sum(-1, keepdims=True) / mask_.sum(-1, keepdims=True))

        pp = self.vparams.values()
        self.cost = (mask_ * ((self.v[:,0] - (rewards_-mean_rewards)) ** 2)).mean()
        self.vgrads = tensor.grad(self.cost, wrt=pp)

class Agent(object):

    def __init__(self, 
                 parent,
                 obs_dim,
                 n_out,
                 out_dim,
                 n_hidden=100,
                 disc_factor=0.,
                 reg_c=0.,
                 activ='tanh',
                 movavg_coeff=0.,
                 vmovavg_coeff=0.,
                 truncate_gradient=-1,
                 optimizer='adam',
                 featex=featex.nothing,
                 featex_args=None,
                 obs_reshape=lambda x, r: s,
                 obs_shape=None):

        self.parent = parent
        self.obs_dim = obs_dim
        self.n_out = n_out
        self.out_dim = out_dim
        self.truncate_gradient = truncate_gradient

        self.n_hidden   = n_hidden
        self.disc_factor= disc_factor
        self.reg_c      = reg_c
        self.activ      = activ
        self.movavg_coeff = movavg_coeff

        self.obs_reshape = obs_reshape
        self.obs_shape = obs_shape

        self.featex = featex
        self.featex_args = featex_args

        self.vars_init()

        self.param_init()
        self.forward_init()

        self.vapprox = VApprox(self, obs_dim, n_hidden=n_hidden, 
                               disc_factor=disc_factor, activ=activ,
                               movavg_coeff=vmovavg_coeff,
                               truncate_gradient=truncate_gradient,
                               featex=featex, featex_args=featex_args)

        self.grad_init()

        self.f_shared, self.f_update = eval(optimizer)(self.params, 
                                                       self.cost,
                                                       self.grads,
                                                       [self.obs, self.actions, 
                                                        self.rewards, self.mask]) 

    def param_init(self):
        self.params = OrderedDict()

        self.params['W'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        for oi in xrange(self.n_out):
            self.params['U%d'%oi] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, self.out_dim[oi]).astype('float32'))
            self.params['c%d'%oi] = theano.shared(numpy.zeros(self.out_dim[oi]).astype('float32'))

        for pp, vv in self.fparams.items():
            self.params[pp] = vv

        if self.movavg_coeff > 0.:
            self.old_params = make_copy(self.params)

    def vars_init(self):
        self.obs = tensor.tensor3('obs', dtype='float32')
        self.actions = tensor.tensor3('act', dtype='int64')
        self.rewards = tensor.matrix('reward', dtype='float32')
        self.mask = tensor.matrix('mask', dtype='float32')

        self.obs_, indim, self.fparams = self.featex(self.obs_reshape(self.obs, 
                                                     self.obs_shape), 
                                                     self.featex_args)
        self.obs_dim = indim if indim is not None else self.obs_dim

    def forward_init(self):
        obs_ = self.obs_.reshape([self.obs_.shape[0]*self.obs_.shape[1], self.obs_.shape[-1]])

        h = eval(self.activ)(tensor.dot(obs_, self.params['W']) + self.params['b'][None,None,:])

        self.pi = []
        for oi in xrange(self.n_out):
            pi = tensor.dot(h, self.params['U%d'%oi]) + self.params['c%d'%oi][None,:]
            pi = tensor.exp(pi - tensor.max(pi,-1,keepdims=True))
            self.pi.append(pi / (pi.sum(-1, keepdims=True)))

        prev = tensor.matrix('prev', dtype='float32')
        #obs = tensor.matrix('obs', dtype='float32')
        obs_ = self.obs_.reshape([self.obs_.shape[0]*self.obs_.shape[1], 
                                  self.obs_.shape[-1]])
        obs_ = obs_[0]

        self.h_init = lambda x: numpy.float32(0.)

        h = eval(self.activ)(tensor.dot(obs_, self.params['W']) + self.params['b'][None,:])

        pi = []
        for oi in xrange(self.n_out):
            pi_ = tensor.dot(h, self.params['U%d'%oi]) + self.params['c%d'%oi][None,:]
            pi_ = tensor.exp(pi_ - tensor.max(pi_,-1,keepdims=True))
            pi.append(pi_ / (pi_.sum(-1, keepdims=True)))

        self.forward = theano.function([self.obs, prev], [h] + pi, name='forward', on_unused_input='ignore')

    def grad_init(self):
        mask_ = self.mask.flatten()
        rewards_ = self.rewards.flatten()
        actions_ = self.actions.reshape([self.actions.shape[0]*self.actions.shape[1],-1])

        #self.mov_std = theano.shared(numpy.float32(1.), 'std')

        pp = self.params.values()
        mean_rewards = (mask_ * rewards_).sum(-1, keepdims=True) / tensor.maximum(1., mask_.sum(-1, keepdims=True))
        centered_rewards = rewards_ - self.vapprox.v[:,0] - mean_rewards
        mean2_rewards = (mask_ * (rewards_ ** 2)).sum(-1, keepdims=True) / tensor.maximum(1., mask_.sum(-1, keepdims=True))
        var_rewards = mean2_rewards - (mean_rewards ** 2)
        scaled_rewards = centered_rewards  / tensor.maximum(1., tensor.sqrt(tensor.maximum(0., var_rewards)))
        #scaled_rewards = centered_rewards

        logprob = 0.
        reg = 0.
        for oi in xrange(self.n_out):
            labs = actions_[:,oi].flatten()
            labs_idx = tensor.arange(labs.shape[0]) * self.out_dim + labs
            logprob = logprob + (mask_ * tensor.log(self.pi[oi].flatten()+1e-6)[labs_idx])
            reg = reg - (self.pi[oi] * tensor.log(self.pi[oi]+1e-6)).sum(-1).sum(0)

        self.cost = -tensor.mean(scaled_rewards * logprob + self.reg_c * reg)
        self.grads = tensor.grad(self.cost, wrt=pp)

    def update(self, obs, acts, rewards, mask):
        self.vapprox.f_shared(obs, rewards, mask)
        self.vapprox.f_update()

        self.f_shared(obs, acts.astype('int64'), rewards, mask)
        self.f_update()

    def sync(self):
        movavg(self.params, self.old_params, self.movavg_coeff)
        transfer(self.old_params, self.params)

        movavg(self.vapprox.vparams, self.vapprox.old_vparams, self.movavg_coeff)
        transfer(self.vapprox.old_vparams, self.vapprox.vparams)


