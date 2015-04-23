import sys, glob, os
import scaleAndMove

def main(argv):
    directory = argv[0]
    os.chdir(directory)
    for file in glob.glob("*.obj"):
        print("modifying mesh " + file)
        scaleAndMove.main([file, os.getcwd()])


if __name__ == "__main__":
    main(sys.argv[1:])
