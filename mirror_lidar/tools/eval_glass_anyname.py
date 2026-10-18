"""Run gsplat/script/eval_synthetic_glass.py on datasets whose image names are not '<frame>_<stamp>.png'
(e.g. rgb_0000.png from the GT-pose COLMAP model of blender_gt_to_colmap.py): frame id = first number in the name.
Same CLI as eval_synthetic_glass.py. Run from ~/SMBC/gsplat/gsplat/script in the gsplat env."""
import os, re, sys
sys.path.insert(0, os.getcwd())
import eval_synthetic_glass as E
E.frame_id = lambda n: int(re.search(r'\d+', os.path.basename(n)).group(0))
E.main()
