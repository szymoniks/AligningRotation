# Run as: blender -b -P <this_script> -- <mesh.obj> <theta> <phi> <resolution_pct> <out_file>
#theta and phi are the spherical coordinates on a sphere of fixed radius 4, of were to position the camera. theta controlsthe north/south axis, phi controls east/west. 0,0 is on top looking down. pi,0 is bottom, looking up.
#resolution percentage is a value 0<x<100 represents what proportion of a full resolution of 1920x1080 were after
import bpy, sys, os, subprocess
from math import cos, sin, pi
import mathutils
C = bpy.context
scene = bpy.context.scene

argv = sys.argv
argv = argv[argv.index("--") + 1:]
fixed_camera_distance = 4.0
def move_camera(theta,phi):
#0,0 is on top looking down.
#pi, 0 is bottom, looking up (i.e. theta controls north/south axis)
    r = fixed_camera_distance
    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)
    C.scene.camera.location = mathutils.Vector((x,y,z))
    
    


full_path = argv[0]
theta = float(argv[1])
phi = float(argv[2])
resolution_pct = int(argv[3])
#out_path = os.getcwd()
out_path = argv[4]
bpy.ops.import_scene.obj(filepath=full_path)
#bpy.ops.object.mode_set(mode='OBJECT')
#print(bpy.data.objects[:])

#bpy.ops.object.select_all(action='DESELECT')
obs = []
for ob in scene.objects:
    if ob.type == 'MESH':
        print('selecting'+ob.name)
        #scene.objects.link(ob)
        ob.select = True
        obs.append(ob)
    else: 
        ob.select = False
#print(C.scene.objects.active)
#print(bpy.data.objects[:])
C.scene.objects.active = obs[0]
bpy.ops.object.join()
print(scene.objects[:])
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

bpy.ops.object.select_all(action='DESELECT')
C.scene.camera.select = True
mesh = list(filter(lambda x: x.type=='MESH', scene.objects[:]))[0]
move_camera(theta,phi)
#C.scene.camera.constraints.new(type='LIMIT_DISTANCE')
#C.scene.camera.constraints["Limit Distance"].target = mesh
#C.scene.camera.constraints["Limit Distance"].distance = fixed_camera_distance

C.scene.camera.constraints.new(type='TRACK_TO')
C.scene.camera.constraints["Track To"].target = list(filter(lambda x: x.type=='MESH', scene.objects[:]))[0]
C.scene.camera.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
C.scene.camera.constraints["Track To"].up_axis = 'UP_Y'
#make background white

bpy.context.scene.world.horizon_color = (1, 1, 1)
bpy.context.scene.render.resolution_percentage = resolution_pct
bpy.context.scene.render.filepath = out_path
#bpy.context.scene.render.filepath = out_path
#bpy.context.scene.render.filepath += "/"+str(full_path.split('/')[-1])

bpy.ops.render.render(write_still=True)

#subprocess.call(["display", bpy.context.scene.render.filepath+".png"])
#subprocess.call(["display", out_path+".png"])

