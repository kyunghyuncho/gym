import logging
import os, sys
import copy

import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy
from collections import OrderedDict

import gym

def sigmoid(x):
    return 1./(1. + numpy.exp(-x))

def grad_clip(grads, t=numpy.float32(1.)):
    gnorm = numpy.float32(0.)
    for gg in grads:
        gnorm = gnorm + (gg ** 2).sum()

    new_grads = []
    for gg in grads:
        new_gg = tensor.switch(tensor.sqrt(gnorm) > t,
                               gg / tensor.sqrt(gnorm) * t,
                               gg)
        new_grads.append(new_gg)

    return new_grads

def weight_decay(params, grads, coeff=numpy.float32(0.001)):
    new_grads = []
    for gg, pp in zip(grads,params.values()):
        new_grads.append(gg + coeff * pp)

    return new_grads

def adam(tparams, grads, inp):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, [], updates=gsup, profile=False, on_unused_input='ignore')

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

# The world's simplest agent!
class BasicAgent(object):

    def __init__(self, 
                 action_space, 
                 obs_space,
                 n_layers=0,
                 n_layers_cr=0,
                 n_hidden=10,
                 disc_factor=1.,
                 n_frames=1,
                 scale=0.1,
                 sigma=0.1,
                 #activ="lambda x: tensor.maximum(x, 0.)",
                 activ="lambda x: tensor.tanh(x)",
                 explo_coeff=0.,
                 epsilon=1e-2,
                 shared=False,
                 mb_size=100,
                 normalize_obs=False):

        print action_space
        print obs_space

        self.mb_size = mb_size
        self.action_space = action_space
        self.obs_space = obs_space
        self.disc_factor = disc_factor # not used yet
        self.n_frames = n_frames
        self.scale = scale if scale > 0. else 1./n_hidden
        self.sigma = numpy.float32(sigma)
        self.explo_coeff = explo_coeff
        self.shared = shared
        self.normalize_obs = normalize_obs
        self.epsilon = epsilon

        self.activ = eval(activ)

        self.n_layers = n_layers
        self.n_layers_cr = n_layers_cr
        self.n_hidden = n_hidden

        self.n_exp = 0
        self.exps = []
        self.scores = []

        self.trng = RandomStreams(1234)

        self.param_init()
        self.tensor_init()

        self.forward_init()
        self.forward_cr_init()

        self.grad_cr_init()
        self.grad_init()

        self.f_cr_shared, self.f_cr_update = adam(self.params_cr, 
                                                  self.grads_cr,
                                                  [self.obs, self.actions, 
                                                   self.rewards, self.r_hat_t1]) 

        self.f_shared, self.f_update = adam(self.params, 
                                            self.grads,
                                            [self.obs, self.actions, self.rewards]) 

    def episode_start(self):
        self.begin = True
        self.last_act = None

        self.n_exp = self.n_exp + 1
        self.exps.append([])
        self.curr_exp = []

    def episode_end(self):
        self.scores.append(numpy.sum([ex[2] for ex in self.exps[-1]]))
        self.curr_exp = None
        self.begin = False

    def record(self, obs, reward):
        obs = self.obs_norm(obs)

        if self.explo_coeff > 0.:
            if len(self.exps[-1]) > 1:
                vel0 = ((self.exps[-1][-1][0] - obs) ** 2).sum()
                reward = reward + self.explo_coeff * vel0

        self.exps[-1].append([obs, self.last_act, reward])
        self.curr_exp.append([obs, self.last_act, reward])

    def count_exps(self):
        return len(self.exps)

    def flush_exps(self, n=-1, worst_first=False):
        if n < 0:
            self.exps = []
            self.scores = []
        elif n <= 0:
            return
        else:
            if worst_first:
                ll = numpy.minimum(n, len(self.exps))
                for ni in xrange(ll):
                    sidx = numpy.argsort(self.scores)
                    del self.exps[sidx[0]]
                    del self.scores[sidx[0]]
            else:
                del self.exps[:n]
                del self.scores[:n]

    def obs_dim(self):
        if type(self.obs_space) == gym.spaces.box.Box:
            return numpy.prod(self.obs_space.shape) 
        else:
            raise Exception('Unsupported observation space')

    def act_dim(self):
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            return env.action_space.n
        elif type(self.action_space) == gym.spaces.Box:
            if len(env.action_space.shape) > 1:
                raise Exception('Only 1-D Box is supported')
            return env.action_space.shape[0]
        else:
            raise Exception('Unsupported action space')

    def param_init(self):
        self.params = OrderedDict()

        self.params['W'] = theano.shared(self.scale * numpy.random.randn(self.obs_dim() * self.n_frames, self.n_hidden).astype('float32'))
        self.params['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        for li in xrange(self.n_layers):
            self.params['W{}'.format(li+1)] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32'))
            self.params['b{}'.format(li+1)] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.params['U'] = theano.shared(self.scale * numpy.random.randn(self.n_hidden, self.act_dim()).astype('float32'))
        self.params['c'] = theano.shared(numpy.zeros(self.act_dim()).astype('float32'))

        # reward predictor
        self.params_cr = OrderedDict()

        if self.shared:
            self.params_cr['W'] = self.params['W']
        else:
            self.params_cr['W'] = theano.shared(self.scale * numpy.random.randn(self.obs_dim() * self.n_frames, self.n_hidden).astype('float32'))
        self.params_cr['V'] = theano.shared(self.scale * numpy.random.randn(self.act_dim(), self.n_hidden).astype('float32'))
        self.params_cr['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        for li in xrange(self.n_layers_cr):
            self.params_cr['W{}'.format(li+1)] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, self.n_hidden).astype('float32'))
            self.params_cr['b{}'.format(li+1)] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.params_cr['U'] = theano.shared(self.scale * numpy.random.randn(self.n_hidden, 1).astype('float32'))
        self.params_cr['c'] = theano.shared(numpy.zeros((1,)).astype('float32'))

    def tensor_init(self):
        self.obs = tensor.matrix('obs', dtype='float32')
        #if type(self.action_space) == gym.spaces.discrete.Discrete:
        #    self.actions = tensor.matrix('act', dtype='int64')
        #elif type(self.action_space) == gym.spaces.Box:
        self.actions = tensor.matrix('act', dtype='float32')
        self.rewards = tensor.vector('reward', dtype='float32')
        self.r_hat_t1 = tensor.vector('r_hat_t+1', dtype='float32')

    def forward_init(self):
        obs = self.obs
        h = tensor.dot(obs, self.params['W']) + self.params['b']
        if type(self.action_space) == gym.spaces.Box:
            h = h + self.sigma * self.trng.normal(size=h.shape, dtype='float32')
        h = self.activ(h)
        for li in xrange(self.n_layers):
            h = tensor.dot(h, self.params['W{}'.format(li+1)]) + self.params['b{}'.format(li+1)]
            if type(self.action_space) == gym.spaces.Box:
                h = h + self.sigma * self.trng.normal(size=h.shape, dtype='float32')
            h = self.activ(h)
        pi = tensor.dot(h, self.params['U']) + self.params['c'][None,:]
        outs = []
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            pi = tensor.exp(pi - tensor.max(pi))
            self.pi = pi / pi.sum(1, keepdims=True)
            outs = [self.pi]
        elif type(self.action_space) == gym.spaces.Box:
            #pi = pi + self.sigma * self.trng.normal(size=pi.shape, dtype='float32')
            pi = tensor.nnet.sigmoid(pi)
            #pi = (tensor.cos(pi) + 1.) / 2.
            #pi = pi + self.sigma * self.trng.normal(size=pi.shape, dtype='float32')
            self.pi = pi
            outs = [self.pi]

        self.forward = theano.function([self.obs], outs, name='forward')

    def grad_init(self):
        pp = self.params.values()
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            self.grads = tensor.grad(-tensor.mean((self.r_hat[:,0][None,:] - self.r_hat.mean()) *
                                                 tensor.log(self.pi)[tensor.arange(self.pi.shape[0]),
                                                     self.actions[:,0]]
                                                 ), 
                                                 wrt=pp)
        elif type(self.action_space) == gym.spaces.Box:
            cost = -tensor.mean(self.r_hat)
            cost = cost + ((self.rewards - self.rewards.mean())[:,None] * ((self.actions - self.pi) ** 2).sum(1)).mean()
            self.grads = tensor.grad(cost, wrt=pp)

        self.grads = weight_decay(self.params, self.grads)
        self.grads = grad_clip(self.grads)

    def forward_cr_init(self):
        obs = self.obs
        h = tensor.dot(obs, self.params_cr['W']) + self.params_cr['b']
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            #h = h + self.params_cr['V'][self.actions[:,0]]
            h = h + tensor.dot(self.pi, self.params_cr['V'])
        elif type(self.action_space) == gym.spaces.Box:
            h = h + tensor.dot(self.pi, self.params_cr['V'])
        h = h + self.sigma * self.trng.normal(size=h.shape, dtype='float32')
        h = self.activ(h)
        for li in xrange(self.n_layers_cr):
            h = tensor.dot(h, self.params_cr['W{}'.format(li+1)]) + self.params_cr['b{}'.format(li+1)]
            h = h + self.sigma * self.trng.normal(size=h.shape, dtype='float32')
            h = self.activ(h)
        self.r_hat = tensor.dot(h, self.params_cr['U']) + self.params_cr['c']
        #self.r_hat = tensor.nnet.sigmoid(self.r_hat)

        self.forward_cr = theano.function([self.obs, self.actions], self.r_hat, name='forward_cr', on_unused_input='ignore')

    def grad_cr_init(self):
        pp = self.params_cr.values()

        self.cost_cr = tensor.mean((self.r_hat - self.disc_factor * self.r_hat_t1[:,None] - self.rewards[:,None]) ** 2)
        self.grads_cr = tensor.grad(self.cost_cr, wrt=pp)

        self.grads_cr = weight_decay(self.params_cr, self.grads_cr)
        self.grads_cr = grad_clip(self.grads_cr)

        self.f_cost_cr = theano.function([self.obs, self.actions, self.rewards, self.r_hat_t1], self.cost_cr, on_unused_input='ignore')

    def collect_minibatch(self, n=100, future=False, accum=True):
        n_obs = numpy.sum([len(ex) for ex in self.exps])

        n = numpy.minimum(n, n_obs)

        # some stupid uniform-random sampling
        obs = numpy.zeros((n, self.n_frames * self.obs_dim())).astype('float32')
        one_obs = numpy.zeros((1, self.n_frames * self.obs_dim())).astype('float32')
        if future:
            obs_t1 = numpy.zeros((n, self.n_frames * self.obs_dim())).astype('float32')
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            acts = numpy.zeros((n, self.act_dim())).astype('int64')
        elif type(self.action_space) == gym.spaces.Box:
            acts = numpy.zeros((n, self.act_dim())).astype('float32')
        rewards = numpy.zeros((n,)).astype('float32')

        #ss = numpy.array(self.scores)
        #pp = numpy.exp(ss) / numpy.exp(ss).sum()

        e_dict = OrderedDict()
        o_dict = OrderedDict()

        for ni in xrange(n):
            e = numpy.random.choice(len(self.exps))
            e_inc = True if e in e_dict else False

            #e = numpy.argmax(numpy.random.multinomial(1, pp))
            if future:
                o = numpy.random.choice(len(self.exps[e])-1)
            else:
                o = numpy.random.choice(len(self.exps[e]))

            o_inc = True if o in o_dict else False

            if e_inc and o_inc:
                continue

            one_obs = one_obs * 0.

            for ii in xrange(self.n_frames):
                try:
                    one_obs[0,ii*self.obs_dim():(ii+1)*self.obs_dim()] = self.exps[e][o-ii][0].squeeze()
                except IndexError:
                    pass

            if ni > 0:
                # try to remove correlated inputs
                min_dist = numpy.min(((obs[:ni,:] - one_obs[0,:][None,:]) ** 2).sum(1))
                if min_dist < self.epsilon:
                    continue
            obs[ni,:] = one_obs

            if future:
                for ii in xrange(self.n_frames):
                    try:
                        obs_t1[ni,ii*self.obs_dim():(ii+1)*self.obs_dim()] = self.exps[e][o-ii+1][0].squeeze()
                    except IndexError:
                        pass
            acts[ni,:] = self.exps[e][o][1]
            if accum:
                rr_ = numpy.array([ex[2] for ex in  self.exps[e][o:]])
                rr_ = rr_ * [self.disc_factor ** ii for ii in xrange(len(rr_))]
                rr_ = rr_.sum()
            else:
                rr_ = self.exps[e][o][2]
            rewards[ni] = rr_

        #print 'average reward', rewards.mean(), 'std', rewards.std()
        if future:
            return obs, obs_t1, acts, rewards
        return obs, acts, rewards

    def update(self):
        obs, acts, rewards = self.collect_minibatch(self.mb_size)

        self.f_shared(obs, acts, rewards)
        self.f_update()

    def update_cr(self):
        obs, obs_t1, acts, rewards = self.collect_minibatch(future=True, accum=False)

        # one-step future
        [a_t1] = self.forward(obs_t1)
        #if type(self.action_space) == gym.spaces.discrete.Discrete:
            #a_t1 = numpy.argmax(a_t1, axis=1)
        a_t1 = a_t1.squeeze().reshape((a_t1.shape[0], self.act_dim()))
        obs_t1 = obs_t1.squeeze().reshape((obs_t1.shape[0], self.obs_dim() * self.n_frames))

        r_hat_t1 = self.forward_cr(obs_t1, a_t1)

        self.f_cr_shared(obs, acts, rewards, r_hat_t1.squeeze())
        self.f_cr_update()

        return self.f_cost_cr(obs, acts, rewards, r_hat_t1.squeeze())

    def obs_norm(self, observation):
        if self.normalize_obs:
            return (observation - self.obs_space.low) / (self.obs_space.high - self.obs_space.low)
        else:
            return observation

    def act(self, observation):
        observation = self.obs_norm(observation)

        obs_win = numpy.zeros([1, self.obs_dim() * self.n_frames]).astype('float32')
        ii = 0
        obs_win[0,ii*self.obs_dim():(ii+1)*self.obs_dim()] = observation.squeeze()
        for ii in xrange(1,self.n_frames-1):
            try: 
                obs_win[0,ii*self.obs_dim():(ii+1)*self.obs_dim()] = self.curr_exp[-ii][0].squeeze()
            except IndexError:
                pass

        outs = self.forward(numpy.float32(obs_win))
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            pi_t = outs[0]
            act = numpy.argmax(numpy.random.multinomial(1, pi_t[0]))
            self.last_act = act
        elif type(self.action_space) == gym.spaces.Box:
            pi = outs[0]
            act = pi[0]
            #act = numpy.maximum(numpy.minimum(act, self.action_space.high), self.action_space.low)
            self.last_act = copy.copy(act)
            act = act * (self.action_space.high - self.action_space.low) + self.action_space.low
            #act = numpy.maximum(numpy.minimum(act, self.action_space.high), self.action_space.low)

        return act

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('Acrobot-v1' if len(sys.argv)<2 else sys.argv[1])
    agent = BasicAgent(env.action_space, env.observation_space, 
            n_hidden=100, n_frames=3, 
            n_layers=0, n_layers_cr=0,
            sigma=.1,
            scale=.01, disc_factor=.9, explo_coeff=0.,
            shared=False,
            mb_size=100)

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True, video_callable=lambda count: count % 50 == 0)

    episode_count = 10000
    max_steps = 1000
    reward = 0
    done = False

    dispFreq = 10
    #flushFreq = 100

    max_exps = 10

    burnin = 0

    updateFreq = 1
    update_steps = 1

    updateCrFreq = 1
    updateCr_steps = 10

    avg_reward = None

    for i in range(episode_count):
        ob = env.reset()

        reward_epi = 0
        penalty_count = 0

        agent.episode_start()
        for j in range(max_steps):
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            if reward < 1e-4:
                penalty_count = penalty_count + 1
            else:
                penalty_count = 0
            #if penalty_count > 100:
            #    reward = -500
            agent.record(ob, reward)
            reward_epi  = reward_epi + reward
            #if penalty_count > 100:
            #    break
            if done:
                break
        agent.episode_end()

        if avg_reward is None:
            avg_reward = reward_epi
        else:
            avg_reward = 0.9 * avg_reward + 0.1 * reward_epi

        if numpy.mod(i+1, dispFreq) == 0:
            print 'Actions: {} ... {}'.format(
                        [e[1][0] for e in agent.exps[-1][:5]],
                        [e[1][0] for e in agent.exps[-1][-5:]])
            print 'Reward at {}-th trial: {}'.format(i, avg_reward)

        if numpy.mod(i+1, updateCrFreq) == 0 and len(agent.exps) > 0:
            #print 'Updating...',
            for j in xrange(updateCr_steps):
                cc = agent.update_cr()
            print 'CR Cost {}'.format(cc)
            #print 'Done'

        if i >= burnin:
            if numpy.mod(i+1, updateFreq) == 0 and len(agent.exps) > 0:
                #print 'Updating...',
                for j in xrange(update_steps):
                    agent.update()
                #print 'Done'

        #if numpy.mod(i, flushFreq) == 0:
        #    print 'Flushing experiences..',
        #    agent.flush_exps()
        #    print 'Done'
        if agent.count_exps() >= max_exps:
            agent.flush_exps(1)

    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    #logger.info("""Successfully ran BasicAgent. Now trying to upload results to 
    #             the scoreboard. If it breaks, you can always just try 
    #             re-uploading the same results.""")
    #gym.upload(outdir, api_key="sk_AyYbf2JmQkmYallZ0QLAg")

