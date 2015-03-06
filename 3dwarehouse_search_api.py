'''
3D Warehouse Search Script - based on original script by Maks Ovsjanikov
'''
from PIL import Image
import sys
import requests
import string
import json
import shutil
import os
import urllib
import urllib2
import zipfile
from StringIO import StringIO

def save_image(image_url, outfile):
    try:
        response = requests.get(image_url)
        img = Image.open(StringIO(response.content))
        img.save(outfile, 'jpeg')
        return 1
    except:
        e = sys.exc_info()[0]
        print "failed to open image " + outfile
        print "error: " + str(e)
        return 0                    


def crop_smontage(file):
    try:
        im = Image.open(str(file))
        outdir = str(file[:-4]) + "_views/"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        bname = os.path.basename(file)
        (w,h) = im.size
        ws = w/36
        for k in range(0, 36):
            box = (k*ws, 0, k*ws+ws, h)
            outfile = file[:-4]+"_view_"+str(k)+".jpg"
            im.crop(box).save(outdir + bname[:-4] + "_view_"+"%02d"%(k)+".jpg","jpeg")
        return 1
    except:
        e = sys.exc_info()[0]
        print "crop swivel error " + file
        print str(e)
        return 0                    

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print "Please specify keyword."
        exit()

    query_string = sys.argv[1]

    count  = 0

    outdir = os.getcwd() + "/" + str.replace(query_string," ","_") + "/"
    TEMP_ZIP = outdir + "temp.zip"

    print(outdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    good_images = []
    model_file = outdir + "models.txt"
    fm = open(model_file, 'w')
    fm.close()
    
    currentMeshId = 0
    
    for skip_results in [0,50,100,150,200]:
        print "Searching for " + query_string
        starti = skip_results + 1
        endi = skip_results + 50
        address = "https://3dwarehouse.sketchup.com/3dw/Search?q="+urllib.quote(query_string)+"&startRow="+str(starti)+"&endRow="+str(endi)
        response = urllib2.urlopen(address)
        data = json.load(response)

        print "found " + str(len(data['entries'])) + " results"
        model_file = outdir + "models.txt"
        fm = open(model_file, 'a')   
        for entry in data['entries']: 
            id = entry['id']
            fm.write(str(id)+"\n")
        fm.close()

        for entry in data['entries']:

            id = entry['id']
            print id
            entity = json.load(urllib2.urlopen("https://3dwarehouse.sketchup.com/3dw/getEntity?id="+id))
            
            if('zip' in entity['binaries']):
                print "LINK " + "https://3dwarehouse.sketchup.com/warehouse/getpubliccontent?contentId=" + id + "&fn=" + TEMP_ZIP
                zipContent = urllib2.urlopen(entity['binaries']['zip']['contentUrl'] + "&fn=" + TEMP_ZIP).read()
                tempZip = open(TEMP_ZIP, "w")
                tempZip.write(zipContent)
                tempZip.close()
                
            
                with zipfile.ZipFile(TEMP_ZIP, "r") as zipFile:
                    os.makedirs( outdir + "mesh" + str(currentMeshId) )
                    zipFile.extractall( outdir + "mesh" + str(currentMeshId) )
                    currentMeshId += 1
                    # for file in zipFile.namelist():
                        # if file.endswith(".dae"):
                            
            

            # outfile = outdir + "entities/" + "img" + "%04d"%(count) + ".txt"
            # fk = open(outfile,'w')
            # fk.write(str(json.dumps(entity, sort_keys=True, indent=4)))
            # fk.close()

            # outfile = outdir + "entities/" + "img" + "%04d"%(count) + "_search_entry.txt"
            # fk = open(outfile,'w')
            # fk.write(str(json.dumps(entry, sort_keys=True, indent=4)))
            # fk.close()

            # if('lt' in entity['binaries']):
                # image_url = entity['binaries']['lt']['url']
                # outfile = outdir + "lt/" + "img" + "%04d"%(count) + ".jpg"
                # save_image(image_url, outfile)

            # if('bot_lt' in entity['binaries']):
                # image_url = entity['binaries']['bot_lt']['url']
                # outfile = outdir + "bot_lt/" + "img" + "%04d"%(count) + ".jpg"
                # save_image(image_url,outfile)

            # if('bot_smontage' in entity['binaries']):
                # image_url = entity['binaries']['bot_smontage']['url']
                # outfile = outdir + "bot_smontage/" + "img" + "%04d"%(count) + ".jpg"
                # if(save_image(image_url,outfile)):
                    # crop_smontage(outfile)

            # if('smontage' in entity['binaries']):
                # image_url = entity['binaries']['smontage']['url']
                # outfile = outdir + "smontage/" + "img" + "%04d"%(count) + ".jpg"
                # if(save_image(image_url,outfile)):
                    # crop_smontage(outfile)

            # count = count + 1
            # print "\r" + str(count)

    fm.close()
