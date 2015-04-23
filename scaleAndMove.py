import sys

from openmesh import *

def main(argv):
    mesh = TriMesh()
    mesh.request_halfedge_normals()
    mesh.request_vertex_normals()
    
    options = Options()
    options += Options.VertexNormal
    
    result = read_mesh(mesh, argv[0], options)
    if not result:
        print "Cannot open the file"
        return 1
    
    
    barycentre = TriMesh.Point(0, 0, 0)
    verticesNumber = 0
    
    print barycentre
        
    for vh in mesh.vertices():
        verticesNumber += 1
        barycentre += mesh.point(vh)
        
    barycentre = barycentre / verticesNumber
    
    
    farthestPoint = TriMesh.Point(0, 0, 0)
    
    print farthestPoint.length()
    
    for vh in mesh.vertices():
        mesh.set_point(vh, mesh.point(vh) - barycentre)
        
        if (  mesh.point(vh).length() > farthestPoint.length() ):
            farthestPoint =  mesh.point(vh)
    
    print farthestPoint.length()    
    
    for vh in mesh.vertices():
        mesh.set_point(vh, mesh.point(vh).normalize() * mesh.point(vh).length() / farthestPoint.length())    
        
    write_mesh(mesh, argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])