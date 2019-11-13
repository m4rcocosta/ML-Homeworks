import os
import shutil
PATH="dataset/"


with open(PATH+"/test/"+"ground_truth.txt","r") as f:
	for r in f:
		splitted = r.split(";")
		img = splitted[0]
		folder = splitted[1].replace(" ","").replace(":","").strip().replace("\r","").replace("\n","")
		#print folder,img
		index=0
		
		if folder=="Mototopo":
		#os.mkdir(PATH+"sc5_mytest/"+folder)
			shutil.copy(PATH+"test/"+img,PATH+"test_bin/"+"Mototopo"+"/"+img)
		else:
			shutil.copy(PATH+"test/"+img,PATH+"test_bin/"+"Others"+"/"+img)

"""
list_imgs=os.listdir(PATH+"sc5_mytest_bin/Mototopo/")
print list_imgs
with open(PATH+"/sc5_test/"+"ground_truth.txt","r") as f:
		for r in f:
			splitted = r.split(";")
			img=splitted[0]
			label=splitted[1]
			if img in list_imgs:
"""				

for folder in os.listdir(PATH+"train/"):
	if "." in folder:
		continue
	l_imgs=os.listdir(PATH+"train/"+folder+"/")
	for img in l_imgs:
		print(img)
		if folder=="Mototopo":
			shutil.copy(PATH+"train/"+folder+"/"+img,PATH+"train_bin"+"/Mototopo")
		else:
			shutil.copy(PATH+"train/"+folder+"/"+img,PATH+"train_bin"+"/Others")

