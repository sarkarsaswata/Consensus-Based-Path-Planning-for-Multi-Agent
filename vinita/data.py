import json
import matplotlib.pyplot as plt
import numpy as np

with open("data4.json", "r") as file:
    data = json.load(file)

time1 = 0
time2 = 0
data1 = []
data2 = []
swapp = []
total = []
for d in data:
    i = 0
    # time1.append(d['t1'][i][1]) #without CA
    time1 = max(sublist[1] for sublist in d['t1'])
    
    time2 = max(sublist[1] for sublist in d['t2'])
    i += 1
    data1.append(time1)
    data2.append(time2)
    swapp.append(d['swappping'])
    total.append(d['col'])
data1 = sorted(data1)
data2 = sorted(data2)


print(sum(swapp) / len(swapp))
print(sum(total) / len(total))

# Calculate mean and standard deviation
mean_data_1 = np.mean(data1)
std_deviation_data_1 = np.std(data1)
mean_data_2 = np.mean(data2)
std_deviation_data_2 = np.std(data2)

# Data for plotting
labels = ['Time wo CA', 'Time with CA']
means = [mean_data_1, mean_data_2]
std_deviations = [std_deviation_data_1, std_deviation_data_2]

# Create a bar plot with error bars
x = np.arange(len(labels))
width = 0.20

fig, ax = plt.subplots()
rects1 = ax.bar(x, means, width, label='Mean')
rects2 = ax.bar(x + width, std_deviations, width, label='Standard Deviation')

ax.set_ylabel('Values')
ax.set_title('Mean and Standard Deviation, 50 agents')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(labels)
ax.legend()

# Add labels with values on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()