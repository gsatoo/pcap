#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-

import sys
import string
import time
from collections import defaultdict
from optparse import OptionParser

import tsv_fields_for_east as tsv_fields
# import tsv_fields as tsv_fields

usage = "usage: %prog [options] keyword"
parser = OptionParser(usage)
parser.add_option('-t', action="store", type="int", dest="flow_timeout", default=120)
parser.add_option('--debug', action="store_true", dest="debug_output", default=False)
(options, args) = parser.parse_args()

def index(field_name):
  return tsv_fields.fields.index(field_name)

def mapper(input):

  TCP = '6'
  UDP = '17'

  i_frame_time_epoch = index("frame.time_epoch")
  i_frame_len = index("frame.len")
  i_ip_src = index("ip.src")
  i_ip_dst = index("ip.dst")
  i_ip_proto = index("ip.proto")
  i_tcp_srcport = index("tcp.srcport")
  i_tcp_dstport = index("tcp.dstport")

  i_tcp_flags_urg = index('tcp.flags.urg')
  i_tcp_flags_ack = index('tcp.flags.ack')
  i_tcp_flags_push = index('tcp.flags.push')
  i_tcp_flags_reset = index('tcp.flags.reset')
  i_tcp_flags_syn = index('tcp.flags.syn')
  i_tcp_flags_fin = index('tcp.flags.fin')
  i_tcp_seq = index('tcp.seq')
  i_tcp_ack = index('tcp.ack')

  def load(input):
    for l in input:
      line = l.strip('\n').split('\t')
      
      time_epoch = line[i_frame_time_epoch]
      len = line[i_frame_len]
      ip_src = line[i_ip_src]
      ip_dst = line[i_ip_dst]
      ip_proto = line[i_ip_proto]

      if ip_proto == TCP:
        srcport = line[i_tcp_srcport]
        dstport = line[i_tcp_dstport]
        
        tcp_flags_urg = line[i_tcp_flags_urg]
        tcp_flags_ack = line[i_tcp_flags_ack]
        tcp_flags_push = line[i_tcp_flags_push]
        tcp_flags_reset = line[i_tcp_flags_reset]
        tcp_flags_syn = line[i_tcp_flags_syn]
        tcp_flags_fin = line[i_tcp_flags_fin]
        tcp_seq = line[i_tcp_seq]
        tcp_ack = line[i_tcp_ack]
        if tcp_ack == '':
          tcp_ack = '-'
        
        direction = '>'
      
        if ip_src > ip_dst:
          ip_temp = ip_src
          ip_src = ip_dst
          ip_dst = ip_temp
          port_temp = srcport
          srcport = dstport
          dstport = port_temp
          direction = '<'
      
        yield ip_src, ip_dst, ip_proto, srcport, dstport, time_epoch, direction, len, tcp_flags_urg, tcp_flags_ack, tcp_flags_push, tcp_flags_reset, tcp_flags_syn, tcp_flags_fin, tcp_seq, tcp_ack

  def dump(records):
    for key in records:
      k = '{} {} {} {} {}'.format(key[0], key[1], key[2], key[3], key[4])
      v = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(key[5], key[6], 1, int(key[7]), key[8], key[9], key[10], key[11], key[12], key[13], key[14], key[15])
      yield '{}\t{}\n'.format(k, v)

  return dump(load(input))

def reducer(input):

  # mapper���o�͂���key, value���p�[�X
  def load(input):
    for l in input:
      k, v = l.strip('\n').split('\t')
      timestamp, direction, packet, byte, urg, ack, psh, rst, syn, fin, seq_no, ack_no = v.split(' ')
      yield k, (timestamp, direction, int(packet), int(byte), urg, ack, psh, rst, syn, fin, seq_no, ack_no)

  # 5tuple�̃^�C���A�E�g����
  def timeout(key, timestamp, ts_begin, ts_last):
    # 5tuple��2�Ԗڈȍ~(ts_begin < timestamp)�̃p�P�b�g�ɂ��āC
    # ���݌��Ă���p�P�b�g�̎������O�񌩂��p�P�b�g�̎�������
    # flow_timeout�b�ȏ�߂��Ă�����ʂ�5tuple�Ƃ��ăJ�E���g�D
    if not ts_begin == 0 and ts_begin < timestamp and options.flow_timeout <= timestamp - ts_last:
      return 1
    else:
      return 0

  def sum(records):

    temp_key = None
    temp_flow_counter = 0
    flow_counter = 0
    ts_begin = 0
    ts_last = 0

    for key, (timestamp, direction, packet, byte, urg, ack, psh, rst, syn, fin, seq_no, ack_no) in sorted(records):

      if not temp_key is None:
        if temp_key == key and timeout(key, float(timestamp), float(ts_begin), float(ts_last)) == 1:
          TCP_CONNECTION = False
          TCP_FINALIZE = 'TIMEOUT'
          flow_counter = temp_flow_counter + 1

      if not temp_key == key or not temp_flow_counter == flow_counter:
        if not temp_key is None:
          yield temp_key, temp_flow_counter, ts_begin, ts_last, total_to_packets, total_to_bytes, total_from_packets, total_from_bytes, TCP_INITIALIZE, TCP_FINALIZE, TCP_RESET

        TCP_CONNECTION = False
        TCP_INITIALIZE = 'CLOSED'
        TCP_FINALIZE = 'LISTEN'
        try_INITIALIZE = 0
        try_FINALIZE = 0
        TCP_RESET = 0

        # 5tuple���ς�����Ƃ��̓t���[�J�E���^�����Z�b�g
        if not temp_key == key:
          temp_key = key
          temp_flow_counter = 0
          flow_counter = 0
        # 5tuple���ς���Ă��Ȃ��Ƃ��̓t���[�J�E���^�������p���D
        else:
          temp_flow_counter = flow_counter
        
        # �g���q�b�N�ʂ����Z�b�g(���Ɖ�������)
        if direction == '>':
          total_to_packets = packet
          total_to_bytes = byte
          total_from_packets = 0
          total_from_bytes = 0
        else:
          total_to_packets = 0
          total_to_bytes = 0
          total_from_packets = packet
          total_from_bytes = byte
        
        ts_begin = timestamp
        ts_last = timestamp
      else:
        # �Z�b�V�������p�����Ă���ꍇ�̓g���q�b�N�ʂ����Z(���Ɖ�������)
        if direction == '>':
          total_to_packets += packet
          total_to_bytes += byte
        else:
          total_from_packets += packet
          total_from_bytes += byte
        ts_last = timestamp

      # TCP�Z�b�V�����J�n����
      if TCP_CONNECTION is False:

        # SYN�p�P�b�g�̌��o
        if TCP_INITIALIZE == 'CLOSED' and syn == '1' and ack == '0':
          TCP_INITIALIZE = 'SYNSENT'

        # SYN/ACK�p�P�b�g�̌��o
        if TCP_INITIALIZE == 'SYNSENT' and syn == '1' and ack == '1':
          TCP_INITIALIZE = 'SYNRCVD'

        # ACK�p�P�b�g�̌��o
        if TCP_INITIALIZE == 'SYNRCVD' and syn == '0' and ack == '1':
          TCP_CONNECTION = True
          TCP_INITIALIZE = 'ESTAB'
          TCP_FINALIZE = 'ESTAB'

        # �Z�b�V�����m���܂łɗv�����p�P�b�g�����J�E���g
        if not TCP_INITIALIZE == 'CLOSED':
          try_INITIALIZE += 1

        # �X�e�[�^�X�̏o��
        if options.debug_output:
          ip_src, ip_dst, ip_proto, srcport, dstport = key.split()
          # print ip_src, srcport, direction, ip_dst, dstport, temp_flow_counter, timestamp, TCP_CONNECTION, TCP_INITIALIZE, 'syn='+syn, 'ack='+ack, 'seq_no='+seq_no, 'ack_no='+ack_no

          print(
            ip_src, 
            srcport, 
            direction, 
            ip_dst, 
            dstport, 
            temp_flow_counter, 
            timestamp, 
            TCP_CONNECTION, 
            TCP_INITIALIZE, 
            'syn='+syn, 
            'ack='+ack, 
            'seq_no='+seq_no, 
            'ack_no='+ack_no
          )

      # TCP�Z�b�V�����I������
      if TCP_CONNECTION is True:

        # FIN�p�P�b�g�̌��o
        if TCP_FINALIZE == 'ESTAB' and fin == '1':
          TCP_FINALIZE = 'FINWAIT'

        # ACK�p�P�b�g�̌��o
        if TCP_FINALIZE == 'FINWAIT' and ack == '1' and fin == '0':
          TCP_FINALIZE = 'TIMEWAIT'
        
        # FIN�p�P�b�g�̌��o
        if (TCP_FINALIZE == 'FINWAIT' or TCP_FINALIZE == 'TIMEWAIT') and fin == '1':
          TCP_FINALIZE = 'CLOSEWAIT'

        # ACK�p�P�b�g�̌��o
        if TCP_FINALIZE == 'CLOSEWAIT' and ack == '1' and fin == '0':
          TCP_CONNECTION = False
          TCP_FINALIZE = 'LASTACK'

        # �Z�b�V�����I���܂łɗv�����p�P�b�g�����J�E���g
        if not TCP_FINALIZE == 0:
          try_FINALIZE += 1

        # �X�e�[�^�X�̏o��
        if options.debug_output:
          ip_src, ip_dst, ip_proto, srcport, dstport = key.split()
          print(
            ip_src, 
            srcport, 
            direction, 
            ip_dst, 
            dstport, 
            temp_flow_counter, 
            timestamp, 
            TCP_CONNECTION, 
            TCP_FINALIZE, 
            'ack='+ack, 
            'fin='+fin, 
            'seq_no='+seq_no, 
            'ack_no='+ack_no
          )

      # TCP���Z�b�g�t���O�̌��o
      if rst == '1':
        TCP_RESET = 'RST'

    if key == temp_key:
      yield temp_key, temp_flow_counter, ts_begin, ts_last, total_to_packets, total_to_bytes, total_from_packets, total_from_bytes, TCP_INITIALIZE, TCP_FINALIZE, TCP_RESET

  def output(data):
    for key in sorted(data):
      k = '{} {}'.format(key[0], key[1])
      v = '{} {} {} {} {} {} {} {} {} {}'.format(key[2], key[3], float(key[3])-float(key[2]), key[4], key[5], key[6] ,key[7], key[8], key[9], key[10])
      yield '{}\t{}\n'.format(k, v)
  
  return output(sum(load(input)))

if __name__ == '__main__':
  taskname = sys.argv[1]
  if taskname == 'mapper':
    sys.stdout.writelines(mapper(sys.stdin))
  elif taskname == 'reducer':
    sys.stdout.writelines(reducer(sys.stdin))