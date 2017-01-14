import logging
import os, sys

import theano
from theano import tensor

import numpy
from collections import OrderedDict

import gym

def adam(tparams, grads, inp):
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

# The world's simplest agent!
class BasicAgent(object):

    def __init__(self, 
                 action_space, 
                 obs_space,
                 n_hidden=10,
                 disc_factor=0.):
        self.action_space = action_space
        self.obs_space = obs_space
        self.disc_factor = disc_factor # not used yet

        self.n_hidden = n_hidden

        self.n_exp = 0
        self.exps = []

        self.param_init()
        self.forward_init()
        self.grad_init()

        self.f_shared, self.f_update = adam(self.params, 
                                            self.grads,
                                            [self.obs, self.actions, self.rewards]) 

        self.f_vshared, self.f_vupdate = adam(self.vparams, 
                                              self.vgrads,
                                              [self.obs, self.rewards]) 

    def episode_start(self):
        self.begin = True
        self.last_act = None

        self.n_exp = self.n_exp + 1
        self.exps.append([])

    def episode_end(self):
        self.begin = False

        # turn reward into future reward
        rlist = numpy.cumsum([e[2] for e in self.exps[-1]][::-1])[::-1]
        for ii, rr in enumerate(rlist):
            self.exps[-1][ii][2] = rr

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

    def act_dim(self):
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            return env.action_space.n
        else:
            raise Exception('Unsupported observation space')

    def param_init(self):
        self.params = OrderedDict()

        self.params['W'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim(), self.n_hidden).astype('float32'))
        self.params['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.params['U'] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, self.act_dim()).astype('float32'))
        self.params['c'] = theano.shared(numpy.zeros(self.act_dim()).astype('float32'))

        self.vparams = OrderedDict()

        self.vparams['W'] = theano.shared(0.1 * numpy.random.randn(self.obs_dim(), self.n_hidden).astype('float32'))
        self.vparams['b'] = theano.shared(numpy.zeros(self.n_hidden).astype('float32'))
        self.vparams['U'] = theano.shared(0.1 * numpy.random.randn(self.n_hidden, 1).astype('float32'))
        self.vparams['c'] = theano.shared(numpy.zeros(1).astype('float32'))

    def forward_init(self):
        self.obs = tensor.matrix('obs', dtype='float32')

        h = tensor.tanh(tensor.dot(self.obs, self.params['W']) + self.params['b'])
        pi = tensor.dot(h, self.params['U']) + self.params['c'][None,:]
        pi = tensor.exp(pi - tensor.max(pi))
        self.pi = pi / pi.sum(1, keepdims=True)

        self.forward = theano.function([self.obs], self.pi, name='forward')

        h = tensor.tanh(tensor.dot(self.obs, self.vparams['W']) + self.vparams['b'])
        self.v = tensor.dot(h, self.vparams['U']) + self.vparams['c'][None,:]

        self.vforward = theano.function([self.obs], self.v, name='vforward')

    def grad_init(self):
        self.actions = tensor.vector('act', dtype='int64')
        self.rewards = tensor.vector('reward', dtype='float32')

        self.mov_std = theano.shared(numpy.float32(1.), 'std')

        pp = self.params.values()
        centered_rewards = self.rewards - self.v[:,0] - self.rewards.mean()
        scaled_rewards = centered_rewards / tensor.maximum(1., self.mov_std)

        self.grads = tensor.grad(-tensor.mean(scaled_rewards * 
                                              tensor.log(self.pi)[tensor.arange(self.pi.shape[0]),
                                                                  self.actions]), 
                                 wrt=pp)

        pp = self.vparams.values()
        self.vgrads = tensor.grad(((self.v[:,0] - (self.rewards - 
                                    self.rewards.mean())) ** 2).mean(), wrt=pp)

    def collect_minibatch(self, n=100):
        n_obs = numpy.sum([len(ex) for ex in self.exps])

        n = numpy.minimum(n, n_obs)

        # some stupid uniform-random sampling
        obs = numpy.zeros((n, self.obs_dim())).astype('float32')
        acts = numpy.zeros((n,)).astype('int64')
        rewards = numpy.zeros((n,)).astype('float32')

        for ni in xrange(n):
            e = numpy.random.choice(len(self.exps))
            o = numpy.random.choice(len(self.exps[e]))
            obs[ni,:] = self.exps[e][o][0]
            acts[ni] = self.exps[e][o][1]
            rewards[ni] = self.exps[e][o][2]

        return obs, acts, rewards

    def update(self, n=100):
        obs, acts, rewards = self.collect_minibatch(n)

        self.mov_std.set_value(numpy.float32(0.8 * self.mov_std.get_value() + 
                                             0.2 * rewards.std()))

        self.f_shared(obs, acts, rewards)
        self.f_update()

    def vupdate(self, n=100):
        obs, acts, rewards = self.collect_minibatch(n)

        self.f_vshared(obs, rewards)
        self.f_vupdate()

    def act(self, observation):
        pi_t = self.forward(numpy.float32(observation.reshape(1,-1)))
        act = numpy.argmax(numpy.random.multinomial(1, pi_t[0]))
        self.last_act = act

        return act

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

    episode_count = 500
    max_steps = 1000
    reward = 0
    done = False

    dispFreq = 100
    flushFreq = 100
    updateFreq = 1
    update_steps = 20

    for i in range(episode_count):
        ob = env.reset()

        reward_epi = 0
        agent.episode_start()
        for j in range(max_steps):
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            agent.record(ob, reward)
            reward_epi  = reward_epi + reward
            if done:
                break
        agent.episode_end()

        if numpy.mod(i, dispFreq) == 0:
            print 'Reward at {}-th trial: {}'.format(i, reward_epi)

        if numpy.mod(i, updateFreq) == 0:
            for j in xrange(update_steps):
                agent.vupdate(128)
            for j in xrange(update_steps):
                agent.update(128)

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

