import sys
import os
import importlib.util

# Standard path setup
ROOT = os.getcwd()

def _import_from_path(module_name, path, extra_paths=[]):
    # Temporarily prepend extra paths
    old_path = sys.path.copy()
    for p in reversed(extra_paths):
        sys.path.insert(0, p)
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = old_path

def train():
    # Keep ROOT in path so train can find local gwen, grpo, etc.
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    import train
    train.main()

def node():
    path = os.path.join(ROOT, "Tome", "mlx-impl", "node.py")
    extra = [os.path.join(ROOT, "Tome", "mlx-impl")]
    module = _import_from_path("tome_node", path, extra_paths=extra)
    module.main()

def proto():
    path = os.path.join(ROOT, "Tome", "mlx-impl", "generate_proto.py")
    extra = [os.path.join(ROOT, "Tome", "mlx-impl")]
    module = _import_from_path("tome_proto", path, extra_paths=extra)
    module.main()

def view():
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    import view_rollouts as vr
    vr.main()

def scheduler():
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    import scripts.scheduler as ss
    ss.main()

def tui():
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    import scripts.tui as st
    st.main()
