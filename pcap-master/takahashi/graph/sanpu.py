import pandas as pd
import matplotlib.pyplot as plt

tcpdf = pd.read_csv("./tcp.csv")
udpdf = pd.read_csv("./udp.csv")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

print(tcpdf[tcpdf["server_port"]<1024]["server_port"].drop_duplicates())
print(udpdf[udpdf["server_port"]<1024]["server_port"].drop_duplicates())

tcpdf[tcpdf["server_port"]==80][["duration","bytes"]].plot(x="duration", y="bytes", kind="scatter", color="orange", ax=ax, label="http(TCP)")
tcpdf[tcpdf["server_port"]==443][["duration","bytes"]].plot(x="duration", y="bytes", kind="scatter", color="blue", ax=ax, label="https(TCP)")
tcpdf[tcpdf["server_port"]==465][["duration","bytes"]].plot(x="duration", y="bytes", kind="scatter", color="red", ax=ax, label="SMTP(TCP)")
tcpdf["duration"], tcpdf[tcpdf["server_port"]==993][["duration","bytes"]].plot(x="duration", y="bytes", kind="scatter", color="pink", ax=ax, label="IMAP(TCP)")
udpdf["duration"], udpdf[udpdf["server_port"]==53][["duration","bytes"]].plot(x="duration", y="bytes", kind="scatter", color="brown", ax=ax, label="DNS(UDP)")
udpdf["duration"], udpdf[udpdf["server_port"]==137 | 138][["duration","bytes"]].plot(x="duration", y="bytes", kind="scatter", color="yellow", ax=ax, label="Net BIOS(UDP)")
udpdf["duration"], udpdf[udpdf["server_port"]==161][["duration","bytes"]].plot(x="duration", y="bytes", kind="scatter", color="green", ax=ax, label="SNMP(UDP)")

plt.savefig("sanpu.png")
plt.close("all")