#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
import gzip
import sys
import os
from functools import partial
# from itertools import imap, ifilterfalse
from subprocess import Popen, PIPE
from multiprocessing import Pool
from collections import defaultdict
from optparse import OptionParser

#import tsvfields
import tsv_fields_for_east as tsv_fields

path_to_tshark = {
# @ubunt is inserted 2017-03-04
	'@mba': '/usr/local/bin/tshark',
	'@ubuntu': '/usr/bin/tshark',
	'@raid04': '/users/tatsuya/local/bin/tshark.old',
	'@raid06': '/usr/local/bin/tshark',
	'@imac':   '/usr/local/bin/tshark', 
	'@raid07': '/users/tatsuya/local/bin/tshark'
# @ubunt is inserted 2017-03-04
#}['@mba']
}['@ubuntu']
# comment out 2017-03-04
#}['@imac']

def include_invalid_field(fields):
	# 不正なフィールド名がないかチェック
	p = Popen([path_to_tshark, '-G'], stdout=PIPE)
	G_fields = {l.decode().strip().split('\t')[2] for l in p.stdout}
	invalid_fields = [f for f in fields if f not in G_fields]
	sys.stderr.writelines("ERROR(Invalid field name): field '{}' not in 'tshark -G'\n".format(f) for f in invalid_fields)
	return len(invalid_fields) != 0

def mktsv(pcap, outdir, fields):
	Popen(['mkdir', '-p', outdir]).communicate()
	output = '{}/{}.tsv.gz'.format(outdir, os.path.basename(pcap))
	cmd = [path_to_tshark, '-nr', pcap, '-Tfields', '-Eseparator=/t', '-Eoccurrence=a'] 
#	 cmd = [path_to_tshark, '-nr', pcap, '-Tfields', '-Eseparator=tab', '-Eoccurrence=a'] 
	cmd.extend(['-e{}'.format(field) for field in fields])
	p1 = Popen(cmd, stdout=PIPE)
	p2 = Popen(['gzip'], stdin=p1.stdout, stdout=open(output, 'wb'))
	p2.communicate()
	
# def tsvloader(input, fields, ignore_empty=False):
# 	def ignore(record):
# 		# まれに tshark が異常終了するなどして、ファイル終端が途中で切れている
# 		# 場合があるので、そのようなレコードを無視する
# 		return len(record) != len(fields)
# 	def dump(record):
# 		if ignore_empty:
# 			return {k:v for k, v in zip(fields, record) if 0 < len(v)}
# 		else:
# 			# 空文字も含めて全キーが存在することを保証する
# 			# キーが完全でないと、get 関数をたくさん書かないといけないのでコードが汚くなる
# 			# 多少効率は悪くなるが、空データも辞書に記録しておく
# 			return dict(zip(fields, record))
# 	records = (l.strip('\n').split('\t') for l in input)
# 	return imap(dump, ifilterfalse(ignore, records))

if __name__ == '__main__':
	fields = tsv_fields.fields
	assert not include_invalid_field(fields)
	
	outdir = sys.argv[1] # tsv 出力先ディレクトリ
	pcaps = sys.argv[2:]
	mapper = partial(mktsv, outdir=outdir, fields=fields)
	
	p = Pool()
	p.imap_unordered(mapper, pcaps)
	p.close()
	p.join()

