import numpy as np
import matplotlib.pyplot as plt
from math import floor

print("Loading data...")

# dataset = "app/final_checks-graph"
# dataset = "nap/msrp1/msrp1-graph"
dataset = "nap/twitterppdb1/twitterppdb1-graph"
N = 100

with open(dataset) as F:
	data = [[v.strip() for v in l.split('\t')] for l in F.readlines()]

header = data.pop(0)
data = {l:{header[i]:float(data[l][i]) for i in range(2, len(data[l]))} for l in range(len(data))} #convert from strings
print(header)
print(data[0])

#bertscore_F1, _recall, _precision
for metric in ["bleurt"]:
	#find min, max values
	vals = [data[i][metric] for i in data]
	(r1,r2) = (min(vals), max(vals))
	step = (r2-r1)/N

	#split data into bins
	(V0,V1,V2) = [[0]*N for i in range(3)]
	lower = 0
	upper = step
	for i in data:
		D = data[i]
		apt = D['apt']
		mi = D['mi']
		binNum = floor((D[metric]-r1)*1.0*(N-1)/(r2-r1))
		if apt:
			V2[binNum]+=1
		elif mi:
			V1[binNum]+=1
		else:
			V0[binNum]+=1
	# biggestBin = max([V0[i]+V1[i]+V2[i] for i in range(N)])
	biggestBin = max(V2)
	
	print(V0, V1, V2)
	#graph data

	ind = np.arange(N)    # the x locations for the groups
	width = 0.8       # the width of the bars: can also be len(x) sequence

	p1 = plt.bar(ind, V2, width, color='tab:green')#, yerr=menStd)
	# p2 = plt.bar(ind, V1, width,
	# 			 bottom=V0)#, yerr=womenStd)
	# p3 = plt.bar(ind, V2, width, bottom=[V0[i]+V1[i] for i in range(N)])

	plt.xlabel(metric)
	plt.title(dataset)# + ' as predicted by ' + metric)
	# plt.xticks(ind, [str(step*i*1.0/N) + "-" + str(step*(i+1)*1.0/N) for i in range(N)])
	xticks = [0,25,50,75,100]
	plt.xticks(xticks, [floor(x*(r2-r1)+100.0*r1)/100 for x in xticks])
	print(r1,r2)
	plt.yticks([floor(biggestBin*i/10) for i in range(10)])
	# plt.legend((p1[0], p2[0], p3[0]), ('non-MI', 'MI', "APT"))

	plt.savefig(dataset+'-apt.png')