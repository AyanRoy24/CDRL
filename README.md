### Modifications :-

- added file [ant_n](examples/customized/mods/offline_safety_gymnasium/ant_n.py) with class for constraints and dynamics modification
- initiated it as safety gym env in [gym_envs](examples/customized/mods/offline_safety_gymnasium/gym_envs.py)
- collected dataset for these mods using [collect_dataset.py](examples/customized/collect_dataset.py)
- notebook for [EDA](logs/fast-safe-rl/AntMod-cost-5-100/eda.ipynb) shows 
  - reward cost distribution for std DSRL SafetyAntVelocityGymnasium dataset 
  - vs [mod1](logs/trpol-75b3) using AntEnv as base class 
  - and [mod2](logs/trpol-2ed8) using SafetyAntVelocityEnv as base class

## 🛠️ Installation

FSRL requires Python >= 3.8. You can install it from source by:
```shell
git clone https://github.com/liuzuxin/fsrl.git
cd fsrl
pip install -e .
```
<!-- It is currently hosted on [PyPI](https://pypi.org/project/fsrl/). 
You can simply install FSRL with the following command:
```shell
pip install fsrl
``` -->

<!-- You can also install with the newest version through GitHub: -->
You can also directly install it with pip through GitHub:
```shell
pip install git+https://github.com/liuzuxin/fsrl.git@main --upgrade
```

You can check whether the installation is successful by:
```python
import fsrl
print(fsrl.__version__)
```

