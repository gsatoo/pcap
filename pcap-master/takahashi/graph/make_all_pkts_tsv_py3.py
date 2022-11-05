# python make_all_pkts_tsv_py3.py [出力先ディレクトリ] [pcapファイル…]
# pcapファイルをtsv_fields_for_east.pyのリストにしたがって集計してtsvにする。(gzで圧縮して指定ディレクトリに出力)
import functools
import subprocess
import os
import sys
import multiprocessing

import tsv_fields_for_east as tsv_fields

# コマンド実行する。そのまま出力されたものを文字列で返す。(エラー時は例外を投げる)
def runcmd(command):
	result = subprocess.run(args=command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if(result.returncode != 0):
		# コマンド実行でエラーがあったとき
		print(result.stderr.decode('utf8'))
		raise Exception("Command execution failed.")
	else:
		return result.stdout.decode('utf8')

# 集計するように定義したフィールド全てがtsharkのフィールドに含まれるかチェックする。NGなら例外を投げる。OKならTrueを返す
def check_field(fields):
	# tshark -G を実行。フィールドを取得。
	result = runcmd("tshark -G")
	# 「定義したフィールド - 取得したフィールド」　定義したフィールドがすべて取得したフィールドに含まれるかどうか調べる
	# (右側)コマンドで取得した結果を行ごとに分け、さらに行の中でタブに分かれている部分から必要な部分(3列目)を取り出し新たな配列にしています
	ngfields = set(fields) - set(map(lambda val: val.split("\t")[2], result.splitlines()))
	if(len(ngfields) != 0):
		raise Exception(f"ERROR(Invalid field name): field {', '.join(ngfields)} not in 'tshark -G'\n")
	return True

#引数誤り防止(意図せぬ動作防止)
def getargs():
	if(len(sys.argv)<3):
		raise Exception("Missing argument")
	outdir = sys.argv[1]
	pcaps = sys.argv[2:]
	if(subprocess.run(f"mkdir -p {outdir}", shell=True, stdout=subprocess.PIPE).returncode != 0):
		raise Exception("The first argument is not a directory.(or failed to create directory.)")
	for pcap in pcaps:
		if(os.path.isfile(pcap)==False):
			raise Exception("Contains non-file after the second argument")
	return {"outdir":outdir, "pcaps": pcaps}

# pcapをtsv(タブ区切り)にしgzで圧縮		pcap:処理するpcapのパス, outdir:出力先ディレクトリ, fields:集計するフィールド
def mktsv(pcap, outdir, fields):
	pcap = os.path.abspath(pcap)	#絶対パスにしておく(事故防止)
	outdir = os.path.abspath(outdir)
	outpath = f"{outdir}/{os.path.basename(pcap)}.tsv"

	runcmd(f"mkdir -p {outdir}")	#出力先ディレクトリがなければ作成
	runcmd(f"tshark -nr {pcap} -T fields -e {' -e '.join(fields)} -E separator=/t -E occurrence=a > {outpath}")
	runcmd(f"gzip -f {outpath}")	#同名ファイルは上書き

if __name__ == '__main__':
	fields = tsv_fields.fields
	check_field(fields)

	args = getargs()

	# 最大4プロセスで並列処理
	wrapper = functools.partial(mktsv, outdir=args["outdir"], fields=fields)
	with multiprocessing.Pool(processes=4) as pool:
		pool.map(wrapper, args["pcaps"])
	#for pcap in args["pcaps"]:
	#	mktsv(pcap, args["outdir"], fields)