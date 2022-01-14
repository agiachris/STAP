from json import load
from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import tempfile
import io
import gym
import collections
import copy
import datetime
import os
import shutil

def save_episode(episode, path):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with open(path, 'wb') as f:
            f.write(bs.read())

def load_episode(path):
    with open(path, 'rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

class ReplayBuffer(torch.utils.data.IterableDataset):
    '''
    This replay buffer is carefully implemented to run efficiently and prevent multiprocessing
    memory leaks and errors.
    All variables starting with an underscore ie _variable are used only by the child processes
    All other variables are used by the parent process.
    '''
    def __init__(self, observation_space, action_space, 
                       discount=0.99, nstep=1, preload_path=None,
                       capacity=100000, fetch_every=1000, cleanup=True, batch_size=None):
        # Observation and action space values
        self.observation_space = observation_space
        self.action_space = action_space

        # Queuing values
        self.discount = discount
        self.nstep = nstep
        self.batch_size = batch_size

        # Data storage values
        self.capacity = capacity
        self.cleanup = cleanup # whether or not to remove loaded episodes from disk
        self.fetch_every = fetch_every

        # worker values to be shared across processes.
        self.preload_path = preload_path
        self.num_episodes = 0
        self.storage_path = tempfile.mkdtemp()
        print("[research] Replay Buffer Storage Path", self.storage_path)

    @property
    def is_parallel(self):
        return not hasattr(self, "is_serial")

    def save(self, path):
        '''
        Save the replay buffer to the specified path. This is literally just copying the files
        from the storage path to the desired path. By default, we will also delete the original files.
        '''
        if self.cleanup:
            print("[research] Warning, attempting to save a cleaned up replay buffer. There are likely no files")
        os.makedirs(path, exist_ok=True)
        srcs = os.listdir(self.storage_path)
        for src in srcs:
            shutil.move(os.path.join(self.storage_path, src), os.path.join(path, src))
        print("Successfully saved", len(srcs), "episodes.")
    
    def __del__(self):
        if not self.cleanup:
            return
        paths = [os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)]
        for path in paths:
            try:
                os.remove(path)
            except:
                pass
        try:
            os.rmdir(self.storage_path)
        except:
            pass

    def _setup_buffers(self):
        self._idx = 0
        self._size = 0
        self._capacity = self.capacity // self._num_workers

        def construct_buffer_helper(space):
            if isinstance(space, gym.spaces.Dict):
                return {k: construct_buffer_helper(v) for k, v in space.items()}
            elif isinstance(space, gym.spaces.Box):
                return np.empty((self._capacity, *space.shape), dtype=space.dtype)
            elif isinstance(space, gym.spaces.Discrete):
                return np.empty((self.capacity,), dtype=np.int64)
            else:
                raise ValueError("Invalid space provided")
        
        self._obs_buffer = construct_buffer_helper(self.observation_space)
        self._action_buffer = construct_buffer_helper(self.action_space)
        self._reward_buffer = np.empty((self._capacity,), dtype=np.float32)
        # TODO: Determine best data types for these buffers.
        self._discount_buffer = np.empty((self._capacity,), dtype=np.float32)
        self._done_buffer = np.empty((self._capacity,), dtype=np.bool_)

    def _add_to_buffer(self, obs, action, reward, done, discount):
        # Can add in batches or serially.
        if isinstance(reward, list) or isinstance(reward, np.ndarray):
            num_to_add = len(reward)
        else:
            num_to_add = 1

        if self._idx + num_to_add > self._capacity:
            # Add all we can at first, then add the rest later
            num_b4_wrap = self._capacity - self._idx
            self._add_to_buffer(obs[:num_b4_wrap], action[:num_b4_wrap], reward[:num_b4_wrap], 
                                done[:num_b4_wrap], discount[:num_b4_wrap])
            self._add_to_buffer(obs[num_b4_wrap:], action[num_b4_wrap:], reward[num_b4_wrap:], 
                                done[num_b4_wrap:], discount[num_b4_wrap:])
        else:
            # Add the transition
            def add_to_buffer_helper(buffer, value):
                if isinstance(buffer, dict):
                    for k, v in buffer.items():
                        add_to_buffer_helper(buffer, value[k])
                elif isinstance(buffer, np.ndarray):
                    buffer[self._idx:self._idx+num_to_add] = value
                else:
                    raise ValueError("Attempted buffer ran out of space!")
            
            add_to_buffer_helper(self._obs_buffer, obs)
            add_to_buffer_helper(self._action_buffer, action)
            add_to_buffer_helper(self._reward_buffer, reward)
            add_to_buffer_helper(self._discount_buffer, discount)
            
            add_to_buffer_helper(self._done_buffer, done)
            self._idx = (self._idx + num_to_add) % self._capacity
            self._size = min(self._size + num_to_add, self._capacity)
    
    def add_to_current_ep(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.add_to_current_ep(key + '_' + k, v)
        else:
            self.current_ep[key].append(value)

    def add(self, obs, action=None, reward=None, done=None, discount=None):
        # Make sure that if we are adding the first transition, it is consistent
        assert (action is None) == (reward is None) == (done is None) == (discount is None)
        if action is None:
            # construct dummy transition
            action = self.action_space.sample()
            reward = 0.0
            done = False
            discount = 1.0
        
        # Case 1: Not Parallel and Cleanup: just add to buffer
        if not self.is_parallel:
            # Deep copy to make sure we don't mess up references.
            self._add_to_buffer(copy.deepcopy(obs), copy.deepcopy(action), reward, done, discount)
            if self.cleanup:
                return # Exit if we clean up and don't save the buffer.

        # If we don't have a current episode list, construct one.
        if not hasattr(self, "current_ep"):
            self.current_ep = collections.defaultdict(list)

        # Add values to the current episode
        self.add_to_current_ep("obs", obs)
        self.add_to_current_ep("action", action)
        self.add_to_current_ep("reward", reward)
        self.add_to_current_ep("done", done)
        self.add_to_current_ep("discount", discount)

        if done:
            # save the episode
            keys = list(self.current_ep.keys())
            assert len(self.current_ep['reward']) == len(self.current_ep['done'])
            obs_keys = [key for key in keys if "obs" in key]
            action_keys = [key for key in keys if "action" in key]
            assert len(obs_keys) > 0, "No observation key"
            assert len(action_keys) > 0, "No action key"
            assert len(self.current_ep[obs_keys[0]]) == len(self.current_ep['reward'])
            # Commit to disk.
            ep_idx = self.num_episodes
            ep_len = len(self.current_ep['reward'])
            episode = {}
            for k, v in self.current_ep.items():
                first_value = v[0]
                if isinstance(first_value, np.ndarray):
                    dtype = first_value.dtype
                elif isinstance(first_value, int):
                    dtype = np.int64
                elif isinstance(first_value, float):
                    dtype = np.float32
                elif isinstance(first_value, bool):
                    dtype = np.bool_
                episode[k] = np.array(v, dtype=dtype)
            # Delete the current_ep reference
            self.current_ep = collections.defaultdict(list)
            # Store the ep
            self.num_episodes += 1
            ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            ep_filename = f'{ts}_{ep_idx}_{ep_len}.npz'
            save_episode(episode, os.path.join(self.storage_path, ep_filename))

    def _load(self, path, cleanup=False):
        ep_filenames = sorted([os.path.join(path, f) for f in os.listdir(path)], reverse=True)
        print(ep_filenames)
        fetched_size = 0
        for ep_filename in ep_filenames:
            ep_idx, ep_len = [int(x) for x in os.path.splitext(ep_filename)[0].split('_')[-2:]]
            if ep_idx % self._num_workers != self._worker_id:
                continue
            if ep_filename in self._episode_filenames:
                break # We found something we have already loaded
            if fetched_size + ep_len > self._capacity:
                break # Cannot fetch more than the size of the replay buffer
            # Load the episode from disk
            try:
                episode = load_episode(ep_filename)
            except:
                continue
            # Add the episode to the buffer
            obs_keys = [key for key in episode.keys() if "obs" in key]
            action_keys = [key for key in episode.keys() if "action" in key]
            obs = {k[len("obs_"):]: episode[k] for k in obs_keys} if len(obs_keys) > 1 else episode[obs_keys[0]]
            action = {k[len("action_"):]: episode[k] for k in action_keys} if len(action_keys) > 1 else episode[action_keys[0]]
            self._add_to_buffer(obs, action, episode["reward"], episode["done"], episode["discount"])
            # maintain the file list and storage
            self._episode_filenames.add(ep_filename)
            if cleanup:
                try:
                    os.remove(ep_filename)
                except OSError:
                    pass

    def _get_one_idx(self):
        # Add 1 for the first dummy transition
        idx = np.random.randint(0, self._size - self.nstep) + 1
        for i in range(self.nstep):
            if self._done_buffer[idx + i - 1]: # We cannot come from a "done" observation, subtract one
                # If the episode is done here, we need to get a new transition!
                return self._get_one_idx() 
        return idx

    def _get_many_idxs(self):
        idxs = np.random.randint(0, self._size - self.nstep, size=int(1.5*self.batch_size)) + 1
        valid = np.ones(idxs.shape, dtype=np.bool_)
        # Mark all the invalid transitions
        for i in range(self.nstep):
            dones = self._done_buffer[idxs + i - 1]  # We cannot come from a "done" observation, subtract one
            valid[dones == True] = False
        valid_idxs = idxs[valid == True] # grab only the idxs that are still valid.
        return valid_idxs[:self.batch_size] # Return the first [:batch_size] of them.

    def _sample(self):
        if self._size <= 1500:
            return {}
        # NOTE: one small bug is that we won't end up being able to sample segments that span
        # Across the barrier. We lose 1 to self.nstep transitions.
        if self.batch_size is None:
            idxs = self._get_one_idx()
        else:
            idxs = self._get_many_idxs()

        obs_idxs = idxs - 1
        next_obs_idxs = idxs + self.nstep - 1

        obs = {k:v[obs_idxs] for k, v in self._obs_buffer} if isinstance(self._obs_buffer, dict) else self._obs_buffer[obs_idxs]
        action = {k:v[idxs] for k, v in self._action_buffer} if isinstance(self._action_buffer, dict) else self._action_buffer[idxs]
        next_obs = {k:v[next_obs_idxs] for k, v in self._obs_buffer} if isinstance(self._obs_buffer, dict) else self._obs_buffer[next_obs_idxs]
        reward = np.zeros_like(self._reward_buffer[idxs])
        discount = np.ones_like(self._discount_buffer[idxs])
        for i in range(self.nstep):
            step_reward = self._reward_buffer[idxs + i]
            reward += discount * step_reward
            discount *= self._discount_buffer[idxs + i] * self.discount
        return dict(obs=obs, action=action, next_obs=next_obs, reward=reward, discount=discount)

    def __iter__(self):
        assert not hasattr(self, "setup"), "Attempted to re-call iter on ReplayBuffer. Should only be called once!"
        self.setup = True
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Be EXTREMELEY careful here to not modify any values that are in the parent object.
            # This is only called if we are in the serial case!
            self.is_serial = True

        # Setup values to be used by this worker in setup
        self._num_workers = worker_info.num_workers if worker_info is not None else 1
        self._worker_id = worker_info.id if worker_info is not None else 0

        self._setup_buffers()

        # setup episode tracker to track loaded episodes
        self._episode_filenames = set()
        self._samples_since_last_load = 0

        if self.preload_path is not None:
            self._load(self.preload_path, cleanup=False) # Load any initial episodes

        while True:
            yield self._sample()
            if self.is_parallel:
                self._samples_since_last_load += 1
                if self._samples_since_last_load >= self.fetch_every:
                    self._load(self.storage_path, cleanup=self.cleanup)
                    self._samples_since_last_load = 0
