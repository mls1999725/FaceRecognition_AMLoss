import os,sys,csv
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt
import numpy as np
plt.switch_backend('agg')

path = '/home/baixy/Datasets/CASIA_aligned'

if not os.path.isdir(path):
    print('error')
else:
    freq = []
    for list in os.listdir(path):
        subpath = os.path.join(path, list)
        i = 0
        for file in os.listdir(subpath):
            i += 1
        freq.append(i)
min_f = min(freq)
max_f = max(freq)
freq.sort(reverse=True)

print(len(freq))
x = np.linspace(1, len(freq), len(freq))


# with open('./sortcasia_webface_analyse.csv', 'w') as file_out:
#     # file_out.write()
#     csv_writer = csv.writer(file_out)
#     for i in freq:
#         csv_writer.writerow([i])

# x = []
# y = []
# for i in range(min_f, max_f, 1):
#     x.append(i)
#     y.append(len([f for f in freq if f >= i]))

plt.plot(x, freq)
plt.xlabel("Person ID")
plt.ylabel("Number of images per person")
plt.savefig("./Figures/analyse_casia_web.png")

# f1 = open('./xcasia_webface_analyse.txt', 'w')
# f1.writelines(str(x))
# f1.close()
# f1 = open('./ycasia_webface_analyse.txt', 'w')
# f1.writelines(str(y))
# f1.close()

# with open('./xcasia_webface_analyse.csv', 'w') as file_out:
#     # file_out.write()
#     csv_writer = csv.writer(file_out)
#     for i in x:
#         csv_writer.writerow([i])
# with open('./ycasia_webface_analyse.csv', 'w') as file_out:
#     # file_out.write()
#     csv_writer = csv.writer(file_out)
#     for i in y:
#         csv_writer.writerow([i])
