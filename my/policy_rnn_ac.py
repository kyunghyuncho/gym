import logging
import os, sys

import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy
from collections import OrderedDict

from utils import *

class QApprox(object):

    def __init__(self, 
                 parent,
                 obs_dim,
                 n_out,
                 out_dim,
                 n_hidden=100,
                 disc_factor=0.,
                 activ='tanh',
                 movavg_coeff=0.,
                 truncate_gradient=-1,
                 optimizer='adam'):

        self.parent = parent
        self.obs_dim = obs_dim
        self.n_out = n_out
        self.out_dim = out_dim
        self.truncate_gradient = truncate_gradient

        self.n_hidden   = n_hidden
        self.disc_factor= disc_factor
        self.activ      = activ
        self.movavg_coeff = movavg_coeff

        self.vars_init()

        self.param_init()
        self.forward_init()
        self.grad_init()

        self.f_shared, self.f_update = eval(optimizer)(self.params, 
                                                       self.cost,
                                                       self.grads,
                                                       [self.obs, self.actions, 
                                                        self.rewards, self.mask]) 

    def param_init(self):
        self.params = OrderedDict()

        self.params['W_init'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['b_init'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))

        self.params['W'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['R'] = theano.shared(0.01 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.params['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.params['Wu'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['Ru'] = theano.shared(0.01 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.params['bu'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.params['Wr'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['Rr'] = theano.shared(0.01 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.params['br'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        for oi in xrange(self.n_out):
            self.params['U%d'%oi] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, self.out_dim[oi]).astype('float32'))
            self.params['c%d'%oi] = theano.shared(numpy.zeros(self.out_dim[oi]).astype('float32'))

        if self.movavg_coeff > 0.:
            self.old_params = make_copy(self.params)

    def vars_init(self):
        self.obs = self.parent.obs
        self.actions = self.parent.actions
        self.rewards = self.parent.rewards
        self.mask = self.parent.mask

    def forward_init(self):
        def _scan(_obs, _h, W, R, b, Wu, Ru, bu, Wr, Rr, br):
            _u = tensor.nnet.sigmoid(tensor.dot(_obs, Wu) + tensor.dot(_h, Ru) + bu[None,:])
            _r = tensor.nnet.sigmoid(tensor.dot(_obs, Wr) + tensor.dot(_h, Rr) + br[None,:])
            h_ = eval(self.activ)(tensor.dot(_obs, W) + _r * tensor.dot(_h, R) + b[None,:])
            h_ = _u * _h + (1. - _u) * h_
            return h_

        h0 = eval(self.activ)(tensor.dot(self.obs[0], self.params['W_init']) + self.params['b_init'])

        h, _ = theano.scan(_scan,
                           sequences=[self.obs],
                           #outputs_info=[tensor.alloc(0., self.obs.shape[1], self.n_hidden)],
                           outputs_info=[h0],
                           non_sequences=[
                               self.params['W'], self.params['R'], self.params['b'],
                               self.params['Wu'], self.params['Ru'], self.params['bu'],
                               self.params['Wr'], self.params['Rr'], self.params['br']
                               ],
                           truncate_gradient=self.truncate_gradient)
        self.Q = []
        for oi in xrange(self.n_out):
            q = tensor.dot(h, self.params['U%d'%oi]) + self.params['c%d'%oi][None,:]
            self.Q.append(q)

    def grad_init(self):
        #self.mov_std = theano.shared(numpy.float32(1.), 'std')

        pp = self.params.values()

        logprob = 0.
        reg = 0.
        for oi in xrange(self.n_out):
            labs = self.actions[:,:,oi].flatten()
            labs_idx = tensor.arange(labs.shape[0]) * self.out_dim + labs
            Qt = self.Q[oi].flatten()[labs_idx].reshape([self.actions.shape[0], 
                                                          self.actions.shape[1]])
            Qt = Qt[:-1]
            Qt1 = self.Q[oi].max(-1)[1:]
            loss = (Qt - (self.rewards[:-1] + (1. - self.disc_factor) * Qt1)) ** 2
            logprob = logprob - (self.mask[:-1] * loss).sum(0)

        self.cost = -tensor.mean(logprob)
        self.grads = tensor.grad(self.cost, wrt=pp)


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
                 optimizer='adam'):

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

        self.vars_init()

        self.qapprox = QApprox(self, obs_dim, n_out, out_dim, n_hidden=n_hidden, 
                               disc_factor=disc_factor, activ=activ,
                               movavg_coeff=vmovavg_coeff,
                               truncate_gradient=truncate_gradient)

        self.param_init()
        self.forward_init()
        self.grad_init()

        self.f_shared, self.f_update = eval(optimizer)(self.params, 
                                                       self.cost,
                                                       self.grads,
                                                       [self.obs, self.actions, 
                                                        self.rewards, self.mask]) 

    def param_init(self):
        self.params = OrderedDict()

        self.params['W_init'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['b_init'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))

        self.params['W'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['R'] = theano.shared(0.01 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.params['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.params['Wu'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['Ru'] = theano.shared(0.01 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.params['bu'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.params['Wr'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim, self.n_hidden).astype('float32'))
        self.params['Rr'] = theano.shared(0.01 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.params['br'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        for oi in xrange(self.n_out):
            self.params['U%d'%oi] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, self.out_dim[oi]).astype('float32'))
            self.params['c%d'%oi] = theano.shared(numpy.zeros(self.out_dim[oi]).astype('float32'))

        if self.movavg_coeff > 0.:
            self.old_params = make_copy(self.params)

    def vars_init(self):
        self.obs = tensor.tensor3('obs', dtype='float32')
        self.actions = tensor.tensor3('act', dtype='int64')
        self.rewards = tensor.matrix('reward', dtype='float32')
        self.mask = tensor.matrix('mask', dtype='float32')

    def forward_init(self):
        def _scan(_obs, _h, W, R, b, Wu, Ru, bu, Wr, Rr, br):
            _u = tensor.nnet.sigmoid(tensor.dot(_obs, Wu) + tensor.dot(_h, Ru) + bu[None,:])
            _r = tensor.nnet.sigmoid(tensor.dot(_obs, Wr) + tensor.dot(_h, Rr) + br[None,:])
            h_ = eval(self.activ)(tensor.dot(_obs, W) + _r * tensor.dot(_h, R) + b[None,:])
            h_ = _u * _h + (1. - _u) * h_
            return h_

        h0 = eval(self.activ)(tensor.dot(self.obs[0], self.params['W_init']) + self.params['b_init'])

        h, _ = theano.scan(_scan,
                           sequences=[self.obs],
                           #outputs_info=[tensor.alloc(0., self.obs.shape[1], self.n_hidden)],
                           outputs_info=[h0],
                           non_sequences=[
                               self.params['W'], self.params['R'], self.params['b'],
                               self.params['Wu'], self.params['Ru'], self.params['bu'],
                               self.params['Wr'], self.params['Rr'], self.params['br']
                               ],
                           truncate_gradient=self.truncate_gradient)
        self.pi = []
        for oi in xrange(self.n_out):
            pi = tensor.dot(h, self.params['U%d'%oi]) + self.params['c%d'%oi][None,:]
            pi = tensor.exp(pi - tensor.max(pi,-1,keepdims=True))
            self.pi.append(pi / (pi.sum(-1, keepdims=True)+1e-6))

        prev = tensor.matrix('prev', dtype='float32')
        obs = tensor.matrix('obs', dtype='float32')

        h0 = eval(self.activ)(tensor.dot(obs, self.params['W_init']) + self.params['b_init'])
        self.h_init = theano.function([obs], h0)

        h = _scan(obs, prev, self.params['W'], self.params['R'], self.params['b'],
                  self.params['Wu'], self.params['Ru'], self.params['bu'],
                  self.params['Wr'], self.params['Rr'], self.params['br'])

        pi = []
        for oi in xrange(self.n_out):
            pi_ = tensor.dot(h, self.params['U%d'%oi]) + self.params['c%d'%oi][None,:]
            pi_ = tensor.exp(pi_ - tensor.max(pi_,-1,keepdims=True))
            pi.append(pi_ / (pi_.sum(-1, keepdims=True)+1e-6))

        self.forward = theano.function([obs, prev], [h] + pi, name='forward')

    def grad_init(self):
        pp = self.params.values()

        logprob = 0.
        reg = 0.
        for oi in xrange(self.n_out):
            labs = self.actions[:,:,oi].flatten()
            labs_idx = tensor.arange(labs.shape[0]) * self.out_dim + labs
            Q_ = self.qapprox.Q[oi].flatten()[labs_idx].reshape([self.actions.shape[0], 
                                                                 self.actions.shape[1]])

            #scaled_rewards = self.rewards - Q_
            scaled_rewards = Q_
            scaled_rewards = scaled_rewards - scaled_rewards.mean()
            scaled_rewards = scaled_rewards / tensor.maximum(1., scaled_rewards.std())

            labs = self.actions[:,:,oi].flatten()
            labs_idx = tensor.arange(labs.shape[0]) * self.out_dim + labs
            logprob = logprob + (scaled_rewards * (self.mask * 
                                  tensor.log(self.pi[oi].flatten()+1e-6)[labs_idx]
                                  .reshape([self.actions.shape[0], 
                                            self.actions.shape[1]]))).sum(0)
            reg = reg - (self.pi[oi] * tensor.log(self.pi[oi]+1e-6)).sum(-1).sum(0)

        self.cost = -tensor.mean(logprob + self.reg_c * reg)
        self.grads = tensor.grad(self.cost, wrt=pp)

    def update(self, obs, acts, rewards, mask):
        cc_q = self.qapprox.f_shared(obs, acts.astype('int64'), rewards, mask)
        self.qapprox.f_update()

        cc = self.f_shared(obs, acts.astype('int64'), rewards, mask)
        self.f_update()

        #print 'Cost {} Q-Cost {}'.format(cc, cc_q)

    def sync(self):
        movavg(self.params, self.old_params, self.movavg_coeff)
        transfer(self.old_params, self.params)

        movavg(self.qapprox.params, self.qapprox.old_params, self.movavg_coeff)
        transfer(self.qapprox.old_params, self.qapprox.params)



