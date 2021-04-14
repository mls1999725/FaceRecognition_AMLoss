import os

# generate label file from dataset
# datasetPath = "/home/baixy/Datasets/CASIA-112x96"
# labelNames = os.listdir(datasetPath)
# labeltxt = open('/home/baixy/Datasets/CASIA_112x96.txt', 'w')
# for label in labelNames:
#     labelpath = os.path.join(datasetPath, label) 
#     imgs = os.listdir(labelpath)
#     for img in imgs:
#         imgpath = os.path.join(labelpath, img)
#         sample = imgpath + ' ' + label + '\n'
#         labeltxt.writelines(sample)

# compute lines
# count = 0
# txt = open("/home/baixy/Datasets/CASIA_origin.txt", 'r')
# for line in txt:
#     count += 1
# print(count)

# split CASIA_origin.txt
# prepath = "/home/baixy/Datasets"
# f = open(os.path.join(prepath, "CASIA_origin_1.txt"), 'w')
# with open(os.path.join(prepath, "CASIA_origin.txt"), 'r') as file:
#     for i, line in enumerate(file.readlines()):
#         if i == 600:
#             break
#         f.writelines(line)

# write to CASIA_maxpy_cleaned.txt
prepath = "/home/baixy/Datasets/CASIA-maxpy-clean"
datapath = "/home/baixy/Datasets/CASIA-maxpy-clean/CASIA-maxpy-clean"
f = open(os.path.join(prepath, "CASIA_maxpy_cleaned_full.txt"), 'w')
with open(os.path.join(prepath, "CASIA_maxpy_cleaned.txt")) as file:
    for i,line in enumerate(file.readlines()):
        item = line.strip().split('\\')
        line =  os.path.join(datapath, item[0], item[1]) + '\n'
        f.writelines(line)

