import sys, glob, os
import scaleAndMove
import math
import subprocess

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
        
def run_command(cmd):
    print "\n\n" + cmd
    subprocess.call(cmd, shell=True)

def main(argv):
#<meshobj> <samples> <outfolder>
    
    fileName = argv[0]
    scaledMeshFileName = fileName + "scaled.obj"
    samplesPerR = argv[1]
    outFolder = argv[2] + "/"
    
    scaleAndMove.main([fileName, scaledMeshFileName])
    
    for x in drange(.001, 1.00001, 1.0/(float(samplesPerR)-1)):
        theta = 2 * math.pi * x - math.pi
        for y in drange(0, 1.00001, 1.0/(float(samplesPerR)-1)):
            phi = math.acos(2*y - 1.0) - math.pi
            run_command("blender -b -P screenshot.py -- " + scaledMeshFileName + " " + str(theta) + " " + str(phi) + " 30 " + outFolder + str(x) + "x" + str(y))
            
    for x in drange(.001, 1.00001, 1.0/(float(samplesPerR)-1)):
        theta = 2 * math.pi * x - math.pi
        for y in drange(0, 1.00001, 1.0/(float(samplesPerR)-1)):
            phi = math.acos(2*y - 1.0) - math.pi
            print "Theta: " + str(theta) + " phi: " + str(phi)
            

if __name__ == "__main__":
    main(sys.argv[1:])
