import sys, glob, os
import scaleAndMove
import math
import subprocess
        
def run_command(cmd):
    print "\n\n" + cmd
    subprocess.call(cmd, shell=True)

def main(argv):
#<meshobj> <samples> <outfolder>
    
    fileName = argv[0]
    scaledMeshFileName = fileName + "scaled.obj"
    samplesPerR = int(argv[1])
    outFolder = argv[2] + "/"
    
    scaleAndMove.main([fileName, scaledMeshFileName])
    
    pi = math.pi
    for x in range(samplesPerR+1)[1:]:
    
        theta = (2*pi) * (float(x)/samplesPerR)
        print x
        print theta
        for y in range(samplesPerR+1)[1:]:
            phi = (2*pi) * (float(y)/samplesPerR)
            run_command("blender -b -P screenshot.py -- " + scaledMeshFileName + " " + str(theta) + " " + str(phi) + " 30 " + outFolder + str(theta) + "x" + str(phi))

            

if __name__ == "__main__":
    main(sys.argv[1:])
