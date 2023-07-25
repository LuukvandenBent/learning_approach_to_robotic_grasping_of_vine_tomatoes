# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['common', 'detect_truss_manual', 'detect_truss_obb', 'determine_grasp_candidates_oriented_keypoint', 'determine_grasp_candidates_manual', 'move_robot', 'pipeline'],
    package_dir={'': 'src'}
)
setup(**setup_args)
