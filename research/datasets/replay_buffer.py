import torch
import numpy as np
import collections
import random
import datetime
import io
import os
import tempfile

class ReplayBuffer(torch.utils.data.IterableDataset):
    '''
    This replay buffer is carefully implemented to run efficiently and prevent multiprocessing
    memory leaks and errors.
    All variables starting with an underscore ie _variable are used only by the child processes
    All other variables are used by the parent process.
    '''
    def __init__(self, observation_space, action_space, discount=0.99, nstep=1, preload_path=None, capacity=100000, fetch_every=1000, cleanup=True):
        # To avoid multiprocessing conflicts, the only values stored here will be constants.
        self.observation_space = observation_space
        self.action_space = action_space
        self.discount = discount
        self.nstep = nstep
        self.capacity = capacity
        self.cleanup = cleanup # whether or not to remove loaded episodes from disk
        self.fetch_every = fetch_every
        
        # Initialize the storage path
        self.preload_path = preload_path
        self.storage_path = tempfile.mkdtemp()
        self.num_episodes = 0
        print("Replay Buffer Storage Path", self.storage_path)

    def add_item(self, key, value):
        if value is None:
            return
        elif isinstance(value, dict):
            for k, v in value.items():
                self.add_item(key + '_' + k, v)
        elif isinstance(value, list):
            self.current_ep[key].extend(value)
        else:
            self.current_ep[key].append(value)

    def add(self, obs, action=None, reward=None, done=None):
        # First check to see if we have created storage buffers.
        # This should only ever execute in the parent and none of the workers
        if not hasattr(self, "current_ep"):
            self.current_ep = collections.defaultdict(list)
     
        self.add_item("obs", obs)
        self.add_item("action", action)
        self.add_item("reward", reward)
        self.add_item("done", done)

        # Check to see if we are done, and if so commit the episode to disk
        if isinstance(done, list):
            done = done[-1]
        if done is not None and done:
            # Run some checks
            keys = list(self.current_ep.keys())
            assert len(self.current_ep['reward']) == len(self.current_ep['done'])
            obs_keys = [key for key in keys if "obs" in key]
            action_keys = [key for key in keys if "action" in key]
            assert len(obs_keys) > 0, "No observation key"
            assert len(action_keys) > 0, "No action key"
            assert len(self.current_ep[obs_keys[0]]) == len(self.current_ep['reward']) + 1
            # Commit to disk.
            # NOTE: Here DRQ-V2 does a cast to the dtype from the observation space.
            episode = {}
            for k, v in self.current_ep.items():
                dtype = v[0].dtype if isinstance(v, np.ndarray) else np.float32
                episode[k] = np.array(v, dtype=dtype)
            # Delete the current_ep reference
            self.current_ep = collections.defaultdict(list)
            # Store the ep
            ep_idx = self.num_episodes
            ep_len = len(self.current_ep['reward'])
            self.num_episodes += 1
            ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            ep_filename = f'{ts}_{ep_idx}_{ep_len}.npz'
            with io.BytesIO() as bs:
                np.savez_compressed(bs, **episode)
                bs.seek(0)
                with open(os.path.join(self.storage_path, ep_filename), 'wb') as f:
                    f.write(bs.read())

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
            os.rename(os.path.join(self.storage_path, src), os.path.join(path, src))
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

    def _load(self, path, cleanup=False):
        ep_filenames = sorted([os.path.join(path, f) for f in os.listdir(path)], reverse=True)
        fetched_size = 0
        for ep_filename in ep_filenames:
            ep_idx, ep_len = [int(x) for x in os.path.splitext(ep_filename)[0].split('_')[-2:]]
            if ep_idx % self._num_workers != self._worker_id:
                continue
            if ep_filename in self._episodes:
                break # We found something we have already loaded
            if fetched_size + ep_len > self.capacity:
                break # Cannot fetch more than the size of the replay buffer
            # load the episode from disk
            with open(ep_filename, 'rb') as f:
                try:
                    raw_episode = np.load(f)
                except:
                    continue # episode failed to move on to the next one.
                episode = {}
                obs_keys = [key for key in raw_episode.keys() if "obs" in key]
                action_keys = [key for key in raw_episode.keys() if "action" in key]
                episode["obs"] = {k[len("obs_"):]: raw_episode[k] for k in obs_keys} if len(obs_keys) > 1 else raw_episode[obs_keys[0]]
                episode["action"] = {k[len("action_"):]: raw_episode[k] for k in action_keys} if len(action_keys) > 1 else raw_episode[action_keys[0]]
                episode["reward"] = raw_episode["reward"]
                episode["discount"] = 1 - raw_episode["done"]
            # Add the length to the fetched size
            fetched_size += ep_len
            # After the episode has been loaded, remove the old episodes
            assert len(episode["reward"]) == ep_len
            while ep_len + self._size > (self.capacity // self._num_workers):
                early_ep_filename = self._episode_filenames.pop(0)
                self._size -= len(self._episodes[early_ep_filename]["reward"])
                del self._episodes[early_ep_filename]
            self._episode_filenames.append(ep_filename)
            self._episode_filenames.sort()
            self._episodes[ep_filename] = episode
            self._size += ep_len
            if cleanup:
                try:
                    os.remove(ep_filename)
                except OSError:
                    pass

    def _sample(self):
        if len(self._episodes) == 0:
            return {}
        episode = self._episodes[random.choice(self._episode_filenames)]
        idx = np.random.randint(0, episode["reward"].shape[0] - self.nstep + 1)
        obs = {k:v[idx] for k, v in episode["obs"].items()} if isinstance(episode["obs"], dict) else episode["obs"][idx]
        action = {k:v[idx] for k, v in episode["action"].items()} if isinstance(episode["action"], dict) else episode["action"][idx]
        next_idx = idx + self.nstep
        next_obs = {k:v[next_idx] for k, v in episode["obs"].items()} if isinstance(episode["obs"], dict) else episode["obs"][next_idx]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self.nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self.discount
        return dict(obs=obs, action=action, next_obs=next_obs, reward=reward, discount=discount)

    def __iter__(self):
        assert not hasattr(self, "current_ep"), "Repaly Buffer worker was reloaded during training!"
        # Setup the workers
        worker_info = torch.utils.data.get_worker_info()
        self._num_workers = worker_info.num_workers if worker_info is not None else 1
        self._worker_id = worker_info.id if worker_info is not None else 0
        self._samples_since_last_load = 0
        self._episode_filenames = []
        self._episodes = {}
        self._size = 0
        if self.preload_path is not None:
            self._load(self.preload_path, cleanup=False) # Load any initial episodes
        while True:
            yield self._sample()
            self._samples_since_last_load += 1
            if self._samples_since_last_load < self.fetch_every:
                self._load(self.storage_path, cleanup=self.cleanup)
                self._samples_since_last_load = 0
