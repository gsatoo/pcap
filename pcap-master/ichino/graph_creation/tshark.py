import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_tcp = pd.read_csv(
    './data/tcp-01-output.tsv',
    delimiter='\t',
    header=None
)
data_udp = pd.read_csv(
    './data/udp-03-output.tsv',
    delimiter='\t',
    header=None
)

csv_array = []
column_array = [
    "項目",
    "平均",
    "中央値",
    "分散",
    "25%点",
    "75%点"
]

def addCsvArray(name, data):
    csv_array.append([
        name,
        np.mean(data),
        np.median(data),
        np.var(data),
        np.percentile(data, q=25),
        np.percentile(data, q=75)
    ])

def portCumulative(port_num, type_name, reverse=False):
    if port_num == "443":
        dic_array = ports_dic_array
    elif port_num == "53":
        dic_array = ports_dic_udp_array

    for port in dic_array:
        if not port["port"] == port_num: continue
        port_array = port[type_name]
        data_x = sorted(port_array, reverse=reverse)
        data_y = [i / len(port_array) for i in range(len(port_array))]

        if reverse == False:
            addCsvArray(type_name + "_" + port_num, port_array)

        if type_name == "traffic":
            plt.xlabel("traffic(byte)")
            plt.ylabel("traffic / length")
        elif type_name == "duration":
            plt.xlabel("duration(sec)")
            plt.ylabel("duration / length")
        plt.plot(data_x, data_y, label=port_num)
        plt.legend()
    if reverse == False:
        file_name = "output/cumulative_" + type_name + "_" +port_num + ".png"
    else:
        file_name = "output/cumulative_reverse_" + type_name + "_" +port_num + ".png"
    plt.savefig(file_name)
    plt.clf()

tcp_array = data_tcp.values
udp_array = data_udp.values

tcp_duration_array = []
tcp_traffic_array = []
tcp_u_traffic_array = []
tcp_u_ports_array = []
tcp_d_traffic_array = []
tcp_d_ports_array = []
udp_duration_array = []
udp_traffic_array = []
udp_u_traffic_array = []
udp_u_ports_array = []
udp_d_traffic_array = []
udp_d_ports_array = []
well_known_array = []
ports_dic_array = []
well_known_udp_array = []
ports_dic_udp_array = []
tcp_packet_size_array = []
udp_packet_size_array = []

# tcpパケット
for tcp in tcp_array:
    n_array = tcp[0].split() + tcp[1].split()
    tcp_duration_array.append(float(n_array[8]))
    tcp_u_traffic_array.append(float(n_array[10]))
    tcp_d_traffic_array.append(float(n_array[12]))
    tcp_traffic_array.append(float(n_array[10]) + float(n_array[12]))
    tcp_u_ports_array.append(float(n_array[3]))
    tcp_d_ports_array.append(float(n_array[4]))

    try:
        tcp_packet_size_array.append(float(n_array[10]) / float(n_array[9]))
        tcp_packet_size_array.append(float(n_array[12]) / float(n_array[11]))
    except ZeroDivisionError:
        print("zero division error")

    if not n_array[4] in well_known_array:
        well_known_array.append(n_array[4])
        ports_dic_array.append({
            "port": n_array[4],
            "duration": [float(n_array[8])],
            "traffic": [float(n_array[10]) + float(n_array[12])],
            "upload": [float(n_array[10])],
            "download": [float(n_array[12])]
        })
    else:
        for port in ports_dic_array:
            if not n_array[4] in port["port"]: continue
            port["duration"].append(float(n_array[8]))
            port["traffic"].append(float(n_array[10]) + float(n_array[12]))
            port["upload"].append(float(n_array[10]))
            port["download"].append(float(n_array[12]))

    if not n_array[3] in well_known_array:
        well_known_array.append(n_array[3])
        ports_dic_array.append({
            "port": n_array[3],
            "duration": [float(n_array[8])],
            "traffic": [float(n_array[10]) + float(n_array[12])],
            "upload": [float(n_array[10])],
            "download": [float(n_array[12])]
        })
    else:
        for port in ports_dic_array:
            if not n_array[3] in port["port"]: continue
            port["duration"].append(float(n_array[8]))
            port["traffic"].append(float(n_array[10]) + float(n_array[12]))
            port["upload"].append(float(n_array[10]))
            port["download"].append(float(n_array[12]))

# udpパケット
for udp in udp_array:
    n_array = udp[0].split() + udp[1].split()
    udp_duration_array.append(float(n_array[8]))
    udp_u_traffic_array.append(float(n_array[10]))
    udp_d_traffic_array.append(float(n_array[12]))
    udp_traffic_array.append(float(n_array[10]) + float(n_array[12]))
    udp_u_ports_array.append(float(n_array[3]))
    udp_d_ports_array.append(float(n_array[4]))

    try:
        udp_packet_size_array.append(float(n_array[10]) / float(n_array[9]))
        udp_packet_size_array.append(float(n_array[12]) / float(n_array[11]))
    except ZeroDivisionError:
        print("zero division error")

    if not n_array[4] in well_known_udp_array:
        well_known_udp_array.append(n_array[4])
        ports_dic_udp_array.append({
            "port": n_array[4],
            "duration": [float(n_array[8])],
            "traffic": [float(n_array[10]) + float(n_array[12])],
            "upload": [float(n_array[10])],
            "download": [float(n_array[12])]
        })
    else:
        for port in ports_dic_udp_array:
            if not n_array[4] in port["port"]: continue
            port["duration"].append(float(n_array[8]))
            port["traffic"].append(float(n_array[10]) + float(n_array[12]))
            port["upload"].append(float(n_array[10]))
            port["download"].append(float(n_array[12]))

    if not n_array[3] in well_known_udp_array:
        well_known_udp_array.append(n_array[3])
        ports_dic_udp_array.append({
            "port": n_array[3],
            "duration": [float(n_array[8])],
            "traffic": [float(n_array[10]) + float(n_array[12])],
            "upload": [float(n_array[10])],
            "download": [float(n_array[12])]
        })
    else:
        for port in ports_dic_udp_array:
            if not n_array[3] in port["port"]: continue
            port["duration"].append(float(n_array[8]))
            port["traffic"].append(float(n_array[10]) + float(n_array[12]))
            port["upload"].append(float(n_array[10]))
            port["download"].append(float(n_array[12]))

# ヒストグラム（継続時間分布, tcp, udp）
plt.hist(
    [
        tcp_duration_array,
        udp_duration_array
    ],
    label=["tcp", "udp"],
    bins=50
)
plt.xlabel("duration(sec)")
plt.ylabel("connections")
plt.legend()
plt.savefig("output/hist_duration.png")
plt.clf()

# ヒストグラム（データ転送量分布, tcp, udp）
plt.hist(
    [
        tcp_traffic_array,
        udp_traffic_array
    ],
    label=["tcp", "udp"],
    bins=50
)
plt.xlabel("traffic(byte)")
plt.ylabel("connections")
plt.legend()
plt.savefig("output/hist_traffic.png")
plt.clf()

# 分布（x: ポート番号, y: データ転送量）
plt.scatter(tcp_u_ports_array, tcp_u_traffic_array, label="tcp(upload)")
plt.scatter(udp_u_ports_array, udp_u_traffic_array, label="udp(upload)")
plt.scatter(tcp_d_ports_array, tcp_d_traffic_array, label="tcp(download)")
plt.scatter(udp_d_ports_array, udp_d_traffic_array, label="udp(download)")
plt.xlabel("ports")
plt.ylabel("traffic(byte)")
plt.legend()
plt.savefig("output/scatter_port_traffic.png")
plt.clf()

# 分布（x: 継続時間, y: データ転送量）
plt.scatter(tcp_duration_array, tcp_traffic_array, label="tcp")
plt.scatter(udp_duration_array, udp_traffic_array, label="udp")
plt.xlabel("duration(sec)")
plt.ylabel("traffic(byte)")
plt.legend()
plt.savefig("output/scatter_duration_traffic.png")
plt.clf()

# 累積分布（データ転送量, 通信種別ごとに総計）
# tcp
tcp_sum_traffic_array = tcp_traffic_array
tcp_traffic_x = sorted(tcp_sum_traffic_array)
tcp_traffic_y = [i / len(tcp_sum_traffic_array) for i in range(len(tcp_sum_traffic_array))]
plt.plot(tcp_traffic_x, tcp_traffic_y, label="tcp")

addCsvArray("traffic_tcp", tcp_traffic_array)

# udp
udp_sum_traffic_array = udp_traffic_array
udp_traffic_x = sorted(udp_sum_traffic_array)
udp_traffic_y = [i / len(udp_sum_traffic_array) for i in range(len(udp_sum_traffic_array))]
plt.xlabel("traffic(byte)")
plt.ylabel("traffic / length")
plt.plot(udp_traffic_x, udp_traffic_y, label="udp")
plt.legend()
plt.savefig("output/cumulative_traffic_tcp_udp.png")
plt.clf()

addCsvArray("traffic_udp", udp_traffic_array)

# 累積分布（継続時間, 通信種別ごとに総計）
# tcp
tcp_duration_x = sorted(tcp_duration_array)
tcp_duration_y = [i / len(tcp_duration_array) for i in range(len(tcp_duration_array))]
plt.plot(tcp_duration_x, tcp_duration_y, label="tcp")

addCsvArray("duration_tcp", tcp_duration_array)

# udp
udp_duration_x = sorted(udp_duration_array)
udp_duration_y = [i / len(udp_duration_array) for i in range(len(udp_duration_array))]
plt.xlabel("duration(sec)")
plt.ylabel("duration / length")
plt.plot(udp_duration_x, udp_duration_y, label="udp")
plt.legend()
plt.savefig("output/cumulative_duration_tcp_udp.png")
plt.clf()

addCsvArray("duration_udp", udp_duration_array)

# 累積分布（データ転送量, ポートごとに）
portCumulative("443", "traffic")
portCumulative("53", "traffic")

# 累積分布（継続時間, ポートごとに）
portCumulative("443", "duration")
portCumulative("53", "duration")

# 補分布（データ転送量, 通信種別ごとに総計）
# tcp
tcp_traffic_x = sorted(tcp_traffic_array, reverse=True)
tcp_traffic_y = [i / len(tcp_traffic_array) for i in range(len(tcp_traffic_array))]
plt.plot(tcp_traffic_x, tcp_traffic_y, label="tcp")

# udp
udp_traffic_x = sorted(udp_traffic_array, reverse=True)
udp_traffic_y = [i / len(udp_traffic_array) for i in range(len(udp_traffic_array))]
plt.xlabel("traffic(byte)")
plt.ylabel("traffic / length")
plt.plot(udp_traffic_x, udp_traffic_y, label="tcp")
plt.legend()
plt.savefig("output/cumulative_reverse_traffic_tcp_udp.png")
plt.clf()

# 補分布(継続時間, 通信種別ごとに総計)
# tcp
tcp_duration_x = sorted(tcp_duration_array, reverse=True)
tcp_duration_y = [i / len(tcp_duration_array) for i in range(len(tcp_duration_array))]
plt.plot(tcp_duration_x, tcp_duration_y, label="tcp")

# udp
udp_duration_x = sorted(udp_duration_array, reverse=True)
udp_duration_y = [i / len(udp_duration_array) for i in range(len(udp_duration_array))]
plt.xlabel("duration(sec)")
plt.ylabel("duration / length")
plt.plot(udp_duration_x, udp_duration_y, label="udp")
plt.legend()
plt.savefig("output/cumulative_reverse_duration_tcp_udp.png")
plt.clf()


# 補分布(データ転送量, ポートごとに)
portCumulative("443", "traffic", reverse=True)
portCumulative("53", "traffic", reverse=True)

# 補分布(継続時間, ポートごとに)
portCumulative("443", "duration", reverse=True)
portCumulative("53", "duration", reverse=True)

addCsvArray("packet_size(tcp)", tcp_packet_size_array)
addCsvArray("packet_size(udp)", udp_packet_size_array)

df = pd.DataFrame(data=csv_array, columns=column_array)
df.to_csv('output/output.csv', encoding='cp932')

