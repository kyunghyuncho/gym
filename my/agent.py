import logging
import os, sys

import theano
from theano import tensor

import numpy
from collections import OrderedDict

import policy_ff
import policy_rnn
import policy_rnn1
import policy_rnn_ac

from utils import *
import gym

# The world's simplest agent!
class MetaAgent(object):

    def __init__(self, 
                 action_space, 
                 obs_space,
                 n_hidden=100,
                 disc_factor=0.,
                 reg_c=0.,
                 n_act_bins=100,
                 n_ens=1,
                 activ='tanh',
                 normalize_obs=False,
                 movavg_coeff=0.,
                 vmovavg_coeff=0.,
                 truncate_gradient=-1,
                 optimizer='adam',
                 agent='policy_ff',
                 max_epi=-1):
        self.action_space = action_space
        self.obs_space = obs_space
        self.disc_factor = disc_factor # not used yet
        self.reg_c = reg_c
        self.activ = activ
        self.normalize_obs = normalize_obs
        self.n_ens = n_ens
        self.movavg_coeff = movavg_coeff
        self.vmovavg_coeff = vmovavg_coeff

        self.max_epi = max_epi

        self.n_hidden = n_hidden
        self.n_act_bins = n_act_bins

        self.n_exp = 0
        self.exps = []

        self.prepare_act_trans()

        obs_dim = self.obs_dim()
        out_dim, n_out = self.act_dim()

        self.agents = []
        for ei in xrange(n_ens):
            self.agents.append(eval('{}.Agent'.format(agent))(self, obs_dim, 
                                     out_dim, n_out, 
                                     n_hidden=n_hidden, disc_factor=disc_factor, 
                                     reg_c=reg_c, activ=activ, 
                                     movavg_coeff=movavg_coeff,
                                     vmovavg_coeff=vmovavg_coeff,
                                     truncate_gradient=truncate_gradient,
                                     optimizer=optimizer))


    def episode_start(self):
        self.begin = True
        self.last_act = None

        self.n_exp = self.n_exp + 1
        self.exps.append([])

        if self.max_epi > 0:
            if self.n_exp > self.max_epi:
                del self.exps[0]
                self.n_exp = self.n_exp - 1

    def episode_end(self):
        self.begin = False

    def record(self, obs, reward):
        self.exps[-1].append([self.obs_norm(obs), self.last_act, reward])

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

    def obs_norm(self, obs):
        if not self.normalize_obs:
            return obs

        if type(self.obs_space) == gym.spaces.box.Box:
            scale = self.obs_space.high - self.obs_space.low
            return (obs - self.obs_space.low) / scale
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
                bins = numpy.linspace(l[0], h[0], self.n_act_bins)

                x_ = numpy.digitize(x, bins) - 1
                return x_

            def trans_box_back(x):
                h = self.action_space.high
                l = self.action_space.low
                bins = numpy.linspace(l[0], h[0], self.n_act_bins)

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

    def collect_minibatch(self, n=100):
        n_out, out_dim = self.act_dim()
        n_obs = len(self.exps)

        n = numpy.minimum(n, n_obs)

        # some stupid uniform-random sampling
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

            discount = numpy.ones(len(exp)) * (1. - self.disc_factor)

            rew = [e[2] for e in exp]
            rr = []
            for ai in xrange(len(exp)):
                frew = rew[ai] + numpy.sum(rew[ai+1:] * numpy.cumprod(discount[ai+1:]))
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
        if len(self.exps) < 2:
            return

        obs, acts, rewards, mask = self.collect_minibatch(n)

        #self.mov_std.set_value(numpy.float32(0.8 * self.mov_std.get_value() + 
        #                                     0.2 * rewards.std()))

        acts = self.act2net(acts)

        for agent in self.agents:
            agent.update(obs, acts, rewards, mask)

    def update_done(self):
        if self.movavg_coeff > 0.:
            for agent in self.agents:
                agent.sync()

    def act(self, observation, prev_h, verbose=False):
        n_out, out_dim = self.act_dim()

        observation = self.obs_norm(observation)

        si = numpy.argmax(numpy.random.multinomial(1, numpy.ones(self.n_ens) / self.n_ens))

        h = []
        pi = 0.
        for ai, agent in enumerate(self.agents):
            pi_t = agent.forward(numpy.float32(observation.reshape(1,-1)),
                                 numpy.float32(prev_h[ai].reshape(1,-1)))

            h.append(pi_t[0])

            if si == ai:
                pi = pi_t[1:]
        #    if ai == 0:
        #        pi = pi_t[1:]
        #    else:
        #        for ii, pp in enumerate(pi_t[1:]):
        #            pi[ii] += pp
        #pi = [pp / self.n_ens for pp in pi]

        act = []
        for oi in xrange(n_out):
            if sum(pi[oi][:-1]) > 1.0:
                pi[oi][:] *= (1. - 1e-4)

            act.append(self.net2act(numpy.argmax(numpy.random.multinomial(1, pi[oi][0]))))

        if numpy.sum(numpy.isnan(pi)) > 0:
            import ipdb; ipdb.set_trace()

        if verbose:
            print pi

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

    #env = gym.make('BipedalWalker-v2' if len(sys.argv)<2 else sys.argv[1])
    env = gym.make('CartPole-v0' if len(sys.argv)<2 else sys.argv[1])
    agent = MetaAgent(env.action_space, env.observation_space, 
                      n_hidden=100, n_ens=1, 
                      reg_c=0.,
                      movavg_coeff=0., vmovavg_coeff=0.,
                      disc_factor=0.5,
                      truncate_gradient=-1,
                      optimizer='adam',
                      agent='policy_rnn1',
                      max_epi=100)

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True)

    episode_count = 20000
    max_steps = 200
    reward_avg = -numpy.Inf
    done = False

    probFreq = -1
    dispFreq = 10
    flushFreq = -1
    updateFreq = 1
    update_steps = 10
    syncFreq = 1
    mb_sz=10

    for i in range(episode_count):
        ob = env.reset()

        reward_epi = 0
        agent.episode_start()
        #prev_h = [numpy.zeros(agent.n_hidden)] * agent.n_ens
        prev_h = []
        for ag in agent.agents:
            prev_h.append(ag.h_init(numpy.float32(ob[None,:])))

        for j in range(max_steps):
            if probFreq > 0 and numpy.mod(j, probFreq) == 0:
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

        if i >= agent.max_epi and numpy.mod(i, updateFreq) == 0:
            for j in xrange(update_steps):
                agent.update(mb_sz)
            agent.update_done()

        if flushFreq > 0 and numpy.mod(i, flushFreq) == 0:
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

