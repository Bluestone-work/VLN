import math,sys
sys.path.insert(0,'research/tools')
from build_directional_causal_pairs import relative_bearing,nearest_sector

def qy(deg):
 a=math.radians(deg)/2; return [0,math.sin(a),0,math.cos(a)]
def check(q,target,want):
 got=relative_bearing({'rotation':q},target,[0,0,0]); assert abs(((got-want+180)%360)-180)<1e-6,(got,want)
check([0,0,0,1],[0,0,-1],0)
check([0,0,0,1],[-1,0,0],90)
check([0,0,0,1],[1,0,0],270)
# 90 degree yaw: local front maps to world left under Habitat convention.
check(qy(90),[-1,0,0],0)
assert nearest_sector(359)==0 and nearest_sector(16)==30 and nearest_sector(345)==0
print('directional bearing tests PASS')
