import mealpy
import inspect

# List all submodules in mealpy
print("Mealpy submodules:")
for name, obj in inspect.getmembers(mealpy):
    if inspect.ismodule(obj):
        print(name)

# Try to find MFO, ZHA, CSA
try:
    from mealpy.swarm_based import MFO
    print("MFO found in swarm_based")
except ImportError:
    print("MFO not found in swarm_based")

try:
    from mealpy.swarm_based import ZHA
    print("ZHA found in swarm_based")
except ImportError:
    print("ZHA not found in swarm_based")

try:
    from mealpy.swarm_based import CSA
    print("CSA found in swarm_based")
except ImportError:
    print("CSA not found in swarm_based")
