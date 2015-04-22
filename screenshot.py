# Run as: blender -b -P <this_script>
import bpy, sys, os


full_path = "/home/md/gfx/AligningRotation/meshes/chair4.obj"
bpy.ops.import_scene.obj(filepath=full_path)
	
bpy.context.scene.render.filepath += '-test'

bpy.ops.render.render(write_still=True)

