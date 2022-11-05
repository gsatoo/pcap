#!/usr/bin/env python3.8
# -*- coding: utf-8 -*- 
import gzip
import sys
import os
from functools import partial
from subprocess import Popen, PIPE
from multiprocessing import Pool
from collections import defaultdict
from optparse import OptionParser

#import tsvfields
import tsv_fields_for_east as tsv_fields

path_to_tshark = {
	'@mba': '/usr/local/bin/tshark',
	'@ubuntu': '/usr/bin/tshark'
}['@ubuntu']

def include_invalid_field(fields):
	# 不正なフィールド名がないかチェック
	p = Popen([path_to_tshark, '-G'], stdout=PIPE)
	G_fields = {l.decode().strip().split('\t')[2] for l in p.stdout}
	invalid_fields = [f for f in fields if f not in G_fields]
	sys.stderr.writelines(
		"ERROR(Invalid field name): field '{}' not in 'tshark -G'\n"
		.format(f) for f in invalid_fields
	)
	return len(invalid_fields) != 0

def mktsv(pcap, outdir, fields):
	Popen(['mkdir', '-p', outdir]).communicate()
	output = '{}/{}.tsv.gz'.format(outdir, os.path.basename(pcap))
	cmd = [path_to_tshark, '-nr', pcap, '-Tfields', '-Eseparator=/t', '-Eoccurrence=a'] 
	cmd.extend(['-e{}'.format(field) for field in fields])
	p1 = Popen(cmd, stdout=PIPE)
	p2 = Popen(['gzip'], stdin=p1.stdout, stdout=open(output, 'wb'))
	p2.communicate()

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
