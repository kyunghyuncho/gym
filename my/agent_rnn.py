import logging
import os, sys

import theano
from theano import tensor

import numpy
from collections import OrderedDict

import gym

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


# The world's simplest agent!
class BasicAgent(object):

    def __init__(self, 
                 action_space, 
                 obs_space,
                 n_hidden=100,
                 disc_factor=0.,
                 reg_c=0.,
                 n_act_bins=100,
                 activ='tanh'):
        self.action_space = action_space
        self.obs_space = obs_space
        self.disc_factor = disc_factor # not used yet
        self.reg_c = reg_c
        self.activ = activ

        self.n_hidden = n_hidden
        self.n_act_bins = n_act_bins

        self.n_exp = 0
        self.exps = []

        self.prepare_act_trans()

        self.param_init()
        self.forward_init()
        self.grad_init()

        self.f_shared, self.f_update = adam(self.params, 
                                            self.grads,
                                            [self.obs, self.actions, 
                                             self.rewards, self.mask]) 

        self.f_vshared, self.f_vupdate = adam(self.vparams, 
                                              self.vgrads,
                                              [self.obs, self.rewards, 
                                               self.mask]) 

    def episode_start(self):
        self.begin = True
        self.last_act = None

        self.n_exp = self.n_exp + 1
        self.exps.append([])

    def episode_end(self):
        self.begin = False

        ## turn reward into future reward
        #rlist = numpy.cumsum([e[2] for e in self.exps[-1]][::-1])[::-1]
        #for ii, rr in enumerate(rlist):
        #    self.exps[-1][ii][2] = rr

    def record(self, obs, reward):
        self.exps[-1].append([obs, self.last_act, reward])

    def flush_exps(self, n=-1):
        if n < 0:
            self.exps = []
        else:
            del self.exps[:n]

    def obs_dim(self):
        if type(self.obs_space) == gym.spaces.box.Box:
            return numpy.prod(self.obs_space.shape) 
        else:
            raise Exception('Unsupported observation space')

    def prepare_act_trans(self):
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            self.act2net = (lambda x: x)
            self.net2act = (lambda x: x)
        elif type(self.action_space) == gym.spaces.Box:
            def trans_box(x):
                h = self.action_space.high
                l = self.action_space.low
                bins = numpy.linspace(l, h, self.n_act_bins)

                x_ = numpy.digitize(x, bins) - 1
                return x_

            def trans_box_back(x):
                h = self.action_space.high
                l = self.action_space.low
                bins = numpy.linspace(l, h, self.n_act_bins)

                return bins[x]

            self.act2net = trans_box
            self.net2act = trans_box_back
        else:
            raise Exception('Unsupported observation space')

    def act_dim(self):
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            return 1, [env.action_space.n]
        if type(self.action_space) == gym.spaces.Box:
            return self.action_space.shape[0], [self.n_act_bins] * self.action_space.shape[0]
        else:
            raise Exception('Unsupported observation space')

    def param_init(self):
        n_out, out_dim = self.act_dim()

        self.params = OrderedDict()

        self.params['W'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim(), self.n_hidden).astype('float32'))
        self.params['R'] = theano.shared(0.1 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.params['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.params['Wu'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim(), self.n_hidden).astype('float32'))
        self.params['Ru'] = theano.shared(0.1 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.params['bu'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        for oi in xrange(n_out):
            self.params['U%d'%oi] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, out_dim[oi]).astype('float32'))
            self.params['c%d'%oi] = theano.shared(numpy.zeros(out_dim[oi]).astype('float32'))

        self.old_params = make_copy(self.params)

        self.vparams = OrderedDict()

        self.vparams['W'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim(), self.n_hidden).astype('float32'))
        self.vparams['R'] = theano.shared(0.1 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.vparams['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.vparams['Wu'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim(), self.n_hidden).astype('float32'))
        self.vparams['Ru'] = theano.shared(0.1 * orth(numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32')))
        self.vparams['bu'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.vparams['U'] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, 1).astype('float32'))
        self.vparams['c'] = theano.shared(numpy.zeros(1).astype('float32'))

        self.old_vparams = make_copy(self.vparams)

    def forward_init(self):
        n_out, out_dim = self.act_dim()

        self.obs = tensor.tensor3('obs', dtype='float32')

        def _scan(_obs, _h, W, R, b, Wu, Ru, bu):
            _u = tensor.nnet.sigmoid(tensor.dot(_obs, Wu) + tensor.dot(_h, Ru) + bu[None,:])
            h_ = eval(self.activ)(tensor.dot(_obs, W) + tensor.dot(_h, R) + b[None,:])
            h_ = _u * _h + (1. - _u) * h_
            return h_

        h, _ = theano.scan(_scan,
                           sequences=[self.obs],
                           outputs_info=[tensor.alloc(0., self.obs.shape[1], self.n_hidden)],
                           non_sequences=[
                               self.params['W'], self.params['R'], self.params['b'],
                               self.params['Wu'], self.params['Ru'], self.params['bu'],
                               ])
        self.pi = []
        for oi in xrange(n_out):
            pi = tensor.dot(h, self.params['U%d'%oi]) + self.params['c%d'%oi][None,:]
            pi = tensor.exp(pi - tensor.max(pi,-1,keepdims=True))
            self.pi.append(pi / pi.sum(-1, keepdims=True))

        prev = tensor.matrix('prev', dtype='float32')
        obs = tensor.matrix('obs', dtype='float32')

        h = _scan(obs, prev, self.params['W'], self.params['R'], self.params['b'],
                  self.params['Wu'], self.params['Ru'], self.params['bu'])

        pi = []
        for oi in xrange(n_out):
            pi_ = tensor.dot(h, self.params['U%d'%oi]) + self.params['c%d'%oi][None,:]
            pi_ = tensor.exp(pi_ - tensor.max(pi_,-1,keepdims=True))
            pi.append(pi_ / pi_.sum(-1, keepdims=True))

        self.forward = theano.function([obs, prev], [h] + pi, name='forward')

        h, _ = theano.scan(_scan,
                        sequences=[self.obs],
                        outputs_info=[tensor.alloc(0., self.obs.shape[1], self.n_hidden)],
                        non_sequences=[
                            self.vparams['W'], self.vparams['R'], self.vparams['b'],
                            self.vparams['Wu'], self.vparams['Ru'], self.vparams['bu']
                            ])
        self.v = tensor.dot(h, self.vparams['U']) + self.vparams['c'][None,:]

        self.vforward = theano.function([self.obs], self.v, name='vforward')

    def grad_init(self):
        n_out, out_dim = self.act_dim()

        self.actions = tensor.tensor3('act', dtype='int64')
        self.rewards = tensor.matrix('reward', dtype='float32')

        self.mask = tensor.matrix('mask', dtype='float32')

        #self.mov_std = theano.shared(numpy.float32(1.), 'std')

        pp = self.params.values()
        mean_rewards = (self.mask * self.rewards).sum(-1, keepdims=True) / self.mask.sum(-1, keepdims=True)
        centered_rewards = self.rewards - self.v[:,:,0] - mean_rewards
        #scaled_rewards = centered_rewards / tensor.maximum(1., centered_rewards.std(-1, keepdims=True))
        scaled_rewards = centered_rewards 

        logprob = 0.
        reg = 0.
        for oi in xrange(n_out):
            labs = self.actions[:,:,oi].flatten()
            labs_idx = tensor.arange(labs.shape[0]) * out_dim + labs
            logprob = logprob + ((self.mask * 
                                  tensor.log(self.pi[oi].flatten())[labs_idx]
                                  .reshape([self.actions.shape[0], 
                                            self.actions.shape[1]])).sum(0))
            reg = reg + (self.pi[oi] * tensor.log(self.pi[oi])).sum(-1).sum(0)

        self.grads = tensor.grad(-tensor.mean(scaled_rewards * logprob + 
                                              self.reg_c * reg), wrt=pp)

        pp = self.vparams.values()
        self.vgrads = tensor.grad((self.mask * 
                                   ((self.v[:,:,0] - (self.rewards - mean_rewards)) ** 2))
                                   .mean(), wrt=pp)

    def collect_minibatch(self, n=100):
        # TODO
        n_out, out_dim = self.act_dim()
        n_obs = numpy.sum([len(ex) for ex in self.exps])

        n = numpy.minimum(n, n_obs)

        # some stupid uniform-random sampling
        obs = numpy.zeros((n, self.obs_dim())).astype('float32')
        acts = numpy.zeros((n,n_out)).astype('float32')
        rewards = numpy.zeros((n,)).astype('float32')

        obs = []
        acts = []
        rewards = []

        min_len = numpy.Inf
        max_len = -numpy.Inf

        for ni in xrange(n):
            e = numpy.random.choice(len(self.exps))
            exp = self.exps[e]

            min_len = numpy.minimum(min_len, len(exp))
            max_len = numpy.maximum(max_len, len(exp))

            obs.append([e[0] for e in exp])
            acts.append([e[1] for e in exp])

            rr = []
            for ai in xrange(len(exp)):
                frew = numpy.sum([((1.-self.disc_factor)**i) * re[2] 
                                  for i, re in enumerate(exp[ai:])])
                rr.append(frew)
            rewards.append(rr)

        min_len = int(min_len)
        max_len = int(max_len)

        obs_tensor = numpy.zeros((max_len, n, self.obs_dim())).astype('float32')
        act_tensor = numpy.zeros((max_len, n, self.act_dim()[0])).astype('float32')
        rew_tensor = numpy.zeros((max_len, n)).astype('float32')
        mask = numpy.zeros((max_len, n)).astype('float32')

        for ni in xrange(n):
            explen = len(rewards[ni])
            obs_tensor[:explen,ni,:] = numpy.concatenate([oo[None,:] for oo in obs[ni][:]], axis=0)
            act_tensor[:explen,ni,:] = numpy.concatenate([numpy.array(oo)[None,:] for oo in acts[ni][:]], axis=0)
            rew_tensor[:explen,ni] = rewards[ni][:]
            mask[:explen,ni] = 1.

        return obs_tensor, act_tensor, rew_tensor, mask

    def update(self, n=100):
        obs, acts, rewards, mask = self.collect_minibatch(n)

        #self.mov_std.set_value(numpy.float32(0.8 * self.mov_std.get_value() + 
        #                                     0.2 * rewards.std()))

        acts = self.act2net(acts)

        self.f_shared(obs, acts.astype('int64'), rewards, mask)
        self.f_update()

    def update_done(self):
        movavg(self.params, self.old_params, 0.95)
        transfer(self.old_params, self.params)

    def vupdate(self, n=100):
        obs, acts, rewards, mask = self.collect_minibatch(n)

        self.f_vshared(obs, rewards, mask)
        self.f_vupdate()

    def vupdate_done(self):
        movavg(self.vparams, self.old_vparams, 0.5)
        transfer(self.old_vparams, self.vparams)

    def act(self, observation, prev_h, verbose=False):
        n_out, out_dim = self.act_dim()

        pi_t = self.forward(numpy.float32(observation.reshape(1,-1)),
                            numpy.float32(prev_h.reshape(1,-1)))

        if verbose:
            print pi_t[1]

        h = pi_t[0]

        act = []
        for oi in xrange(n_out):
            if sum(pi_t[1][oi,:-1]) > 1.0:
                pi_t[1][oi,:] *= (1. - 1e-6)

            act.append(self.net2act(numpy.argmax(numpy.random.multinomial(1, pi_t[1][oi,:]))))

        if verbose:
            print act

        self.last_act = act

        if len(act) == 1:
            act = act[0]

        return act, h

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('CartPole-v0' if len(sys.argv)<2 else sys.argv[1])
    agent = BasicAgent(env.action_space, env.observation_space)

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True)

    episode_count = 2000
    max_steps = 2000
    reward_avg = -numpy.Inf
    done = False

    probFreq = 1000
    dispFreq = 10
    flushFreq = 10
    updateFreq = 1
    update_steps = 10
    mb_sz=16

    for i in range(episode_count):
        ob = env.reset()

        reward_epi = 0
        agent.episode_start()
        prev_h = numpy.zeros(agent.n_hidden)

        for j in range(max_steps):
            if numpy.mod(j, probFreq) == 0:
                action, prev_h = agent.act(ob, prev_h, True)
            else:
                action, prev_h = agent.act(ob, prev_h, False)
            ob, reward, done, _ = env.step(numpy.array(action))
            agent.record(ob, reward)
            reward_epi  = reward_epi + reward
            if done:
                break
        agent.episode_end()

        if reward_avg == -numpy.Inf:
            reward_avg = reward_epi
        else:
            reward_avg = 0.9 * reward_avg + 0.1 * reward_epi

        if numpy.mod(i, dispFreq) == 0:
            print 'Reward at {}-th trial: {}, {}'.format(i, reward_epi, reward_avg)

        if numpy.mod(i, updateFreq) == 0:
            for j in xrange(update_steps):
                agent.vupdate(mb_sz)
            agent.vupdate_done()
            for j in xrange(update_steps):
                agent.update(mb_sz)
            agent.update_done()

        if numpy.mod(i, flushFreq) == 0:
            print 'Flushing experiences..',
            agent.flush_exps()
            print 'Done'

    # Dump result info to disk
    env.monitor.close()

    ## Upload to the scoreboard. We could also do this from another
    ## process if we wanted.
    #logger.info("""Successfully ran BasicAgent. Now trying to upload results to 
    #             the scoreboard. If it breaks, you can always just try 
    #             re-uploading the same results.""")
    #gym.upload(outdir, api_key="sk_AyYbf2JmQkmYallZ0QLAg")

