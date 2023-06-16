## Advanced Usage

#### Factories
The base classes and subclasses of all system components are contained to their own submodule, e.g., `temporal_policies/trainers/`. 
Moreover, every submodule implements a [Factory](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/utils/configs.py#L143) in `temporal_policies/<submodule_name>/utils.py` which parses a configuration file and optional arguments, such as a checkpoint path, and returns the instantiated class. 

This allows for easily loading subsets of TAPS components for a specific training and evaluation script. 
As examples, [loading an AgentTrainer](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/trainers/utils.py#L60) requires [loading an Env](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/envs/utils.py#L8), and [loading a SCODTrainer](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/trainers/utils.py#L123) requires [loading an Agent](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/agents/utils.py#L18). 

### Environments
We implement two environments, [PyBox2D](https://github.com/agiachris/temporal-policies/tree/main/temporal_policies/envs/pybox2d) and [PyBullet](https://github.com/agiachris/temporal-policies/tree/main/temporal_policies/envs/pybullet), the former of which is untested for backwards compatibility after extensive changes were made to the code.

The [TableEnv](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/envs/pybullet/table_env.py#L94) class is used to train all robotic manipulation primitives and evaluate long-horizon plans under domain randomization.
The environment is instantiated according to a configuration file with several key fields.
We will use [pick.yaml](https://github.com/agiachris/temporal-policies/blob/main/configs/pybullet/envs/official/primitives/pick.yaml) as a running example.

#### Primitives
The config first specifies a `pick` under the `primitives:` field.

This field is used to indicate what primitive(s) the environment will use, which determines the action space exposed when training reinforcement learning agents and dynamics models. 
Each primitive comes with a [`PrimitiveAction`](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/envs/pybullet/table/primitive_actions.py#L6) and [`Primitive`](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/envs/pybullet/table/primitives.py#L99) implementation. 
- `PrimitiveAction` specifies the skill parameters, their upper and lower bounds, which are used to rescale policy predicted skill parameters from [0,1] to real-world scale;
- `Primitive` implements functions for primitive execution (with collision checking), action sampling where sampled actions are scenario-agnostic and thus prone to failure, and obtaining the [object argument indices](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/envs/pybullet/table/primitives.py#L135) used to arrange the environment state vector such that target objects are ordered first and the remaining objects are shuffled. 

For reference, see the interplay between [`PickAction`](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/envs/pybullet/table/primitive_actions.py#L32) and [`Pick`](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/envs/pybullet/table/primitives.py#L240) in the context of `pick`.

#### Tasks
The config then specifies an `action_skeleton:` and `initial_state:` under the `tasks:` field. 

Tasks are sampled at uniform random, i.e, for `pick`, the environment alternates between training the agent to either pick up a randomly selected block or the hook.

##### Predicates
Initial states of the environment are sampled so as to satisfy the task's [Predicates](https://github.com/agiachris/temporal-policies/blob/main/temporal_policies/envs/pybullet/table/predicates.py#L22).

As shown for `pick`, the initial state indicates that all objects must be placed `On(Predicate)` the table or rack, and that the rack is `Aligned(Predicate)`.
Note that when `.reset()` is called on `TableEnv`, the environment state will be continually sampled until one that [satisfies all predicates](https://github.com/agiachris/temporal-policies/blob/9b6e51814715f56dc2c286eb550faec873e0cef3/temporal_policies/envs/pybullet/table_env.py#L647) is found. 

#### Objects
The configuration file lastly specifies `objects:`. 

This field indicates a list of [Objects](https://github.com/agiachris/temporal-policies/blob/9b6e51814715f56dc2c286eb550faec873e0cef3/temporal_policies/envs/pybullet/table/objects.py) to be spawned during training or evaluation. 
An important object subclass is the [`Variant(WrapperObject)`](https://github.com/agiachris/temporal-policies/blob/9b6e51814715f56dc2c286eb550faec873e0cef3/temporal_policies/envs/pybullet/table/objects.py#L785) which is applied over an [`ObjectGroup`](https://github.com/agiachris/temporal-policies/blob/9b6e51814715f56dc2c286eb550faec873e0cef3/temporal_policies/envs/pybullet/table/objects.py#L632).
This class is used to sample *one-of-n* possible objects instead of representing a concrete one.
The [specification](https://github.com/agiachris/temporal-policies/blob/main/configs/pybullet/envs/official/primitives/pick.yaml#L44) in `pick` indicates that a sampled environment state may contain between 0-4 blocks.