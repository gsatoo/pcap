import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./tcp.csv")
data = df["duration"]

fig, ax1 = plt.subplots()
data.hist(bins=30, ax=ax1)
plt.xlabel("duration")
plt.axvline(data.mean(), color='black', linestyle='dashed', linewidth=1)
plt.axvline(data.median(), color='black', linestyle='dashed', linewidth=1)
_, max_ylim = plt.ylim()
_, max_xlim = plt.xlim()
plt.text(data.mean()*1.1, max_ylim*0.95, 'Mean: {:.2f}'.format(data.mean()), color='black')
plt.text(data.median()*1.1, max_ylim*1.01, 'Median: {:.2f}'.format(data.median()), color='black')
plt.text(max_xlim*0.6, max_ylim*0.95, 'Variance: {:.2f}'.format(data.var()), color='black')

ax2 = ax1.twinx()
per = np.arange(0.00,1.01,0.01)
ax2.plot(data.quantile(per), per, color="crimson")
ax2.plot(data.quantile(0.25), 0.25, marker="x",color="crimson", markersize=10)
ax2.plot(data.quantile(0.75), 0.75, marker="x",color="crimson", markersize=10)

plt.savefig("tcp_duration.png")
plt.close("all")