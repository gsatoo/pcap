# batに記載があった処理
rm ./out-test/*

python make_all_pkts_tsv_v2.py out-test $*

zcat out-test/* | python mr_tcp_separator.py mapper | sort | python mr_tcp_separator.py reducer | sort -n -k7 > out-data/tcp-01-output.tsv

rm ./out-test/*

python make_all_pkts_tsv_v2.py out-test $*

zcat out-test/* | python mr_tcp_separator-02.py mapper | sort | python mr_tcp_separator-02.py reducer | sort -n -k7 > out-data/tcp-02-output.tsv

rm ./out-test/*

python make_all_pkts_tsv_v2.py out-test $*

zcat out-test/* | python mr_udp_separator-03.py mapper | sort | python mr_udp_separator-03.py reducer | sort -n -k7 > out-data/udp-03-output.tsv

# 課題用に自分で追加

# python make_all_pkts_tsv_v2.py out-test $1

# zcat out-test/$1.tsv.gz | python mr_tcp_separator-03.py mapper | sort | python mr_tcp_separator-03.py reducer | sort -n -k7 > out-test/$1-tcp-02-output.tsv

# batに記載があった未処理分

# python2.7 make_all_pkts_tsv_v2.py out-test data170303

# gzcat out-test/data170303.tsv.gz | python2.7 mr_tcp_separator-02.py mapper | sort| python2.7 mr_tcp_separator-02.py reducer | sort -n -k7


# python2.7 make_all_pkts_tsv_v2.py out-test data-mac-0304
# gzcat out-test/data-mac-0304.tsv.gz | python mr_tcp_separator-02.py mapper | sort | python mr_tcp_separator-02.py reducer  | sort -n -k7 > data-mac-0304-TcpSynFin
# cat -n data-mac-0304-TcpSynFin > tmp

# gzcat out-test/data-mac-0304.tsv.gz | python mr_tcp_separator-03.py mapper | sort | python mr_tcp_separator-03.py reducer  > data-mac-0304-TcpSynFin

# gzcat out-test/data-mac-0304.tsv.gz | python mr_udp_separator-03.py mapper | sort | python mr_udp_separator-03.py reducer  > data-mac-0304-Udp
