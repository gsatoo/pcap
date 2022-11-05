#!/usr/local/bin/python3.8

import sys
import string
import time
from collections import defaultdict
from optparse import OptionParser

import tsv_fields_for_east as tsv_fields

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
  i_udp_srcport = index("udp.srcport")
  i_udp_dstport = index("udp.dstport")

  i_udp_qry_name = index("dns.qry.name")
  i_udp_resp_name = index("dns.resp.name")
  
  def load(input):
    for l in input:
      line = l.strip('\n').split('\t')
      
      time_epoch = line[i_frame_time_epoch]
      len = line[i_frame_len]
      ip_src = line[i_ip_src]
      ip_dst = line[i_ip_dst]
      ip_proto = line[i_ip_proto]

      if ip_proto == UDP:
        srcport = line[i_udp_srcport]
        dstport = line[i_udp_dstport]
        
        direction = '>'
      
        if ip_src > ip_dst:
          ip_temp = ip_src
          ip_src = ip_dst
          ip_dst = ip_temp
          port_temp = srcport
          srcport = dstport
          dstport = port_temp
          direction = '<'
      
        yield ip_src, ip_dst, ip_proto, srcport, dstport, time_epoch, direction, len

  def dump(records):
    for key in records:
      k = '{} {} {} {} {}'.format(key[0], key[1], key[2], key[3], key[4])
      v = '{} {} {} {}'.format(key[5], key[6], 1, int(key[7]))
      yield '{}\t{}\n'.format(k, v)

  return dump(load(input))

def reducer(input):

  def load(input):
    for l in input:
      k, v = l.strip('\n').split('\t')
      timestamp, direction, packet, byte = v.split(' ')
      yield k, (timestamp, direction, int(packet), int(byte))

  def timeout(key, timestamp, ts_begin, ts_last):
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

    for key, (timestamp, direction, packet, byte) in sorted(records):

      if not temp_key is None:
        if temp_key == key and timeout(key, float(timestamp), float(ts_begin), float(ts_last)) == 1:
          flow_counter = temp_flow_counter + 1

      if not temp_key == key or not temp_flow_counter == flow_counter:
        if not temp_key is None:
          if IsFlip == "yes":
            ip_src, ip_dst, ip_proto, srcport, dstport = key.split()
            ip_temp = ip_src
            ip_src = ip_dst
            ip_dst = ip_temp
            port_temp = srcport
            srcport = dstport
            dstport = port_temp
            bytes_temp = total_to_bytes
            total_to_bytes = total_from_bytes
            total_from_bytes = bytes_temp
            packets_temp = total_to_packets
            total_to_packets = total_from_packets
            total_from_packets = packets_temp
            temp_key = '{} {} {} {} {}'.format(ip_src,ip_dst,ip_proto,srcport,dstport)
          
          yield temp_key, temp_flow_counter, ts_begin, ts_last, total_to_packets, total_to_bytes, total_from_packets, total_from_bytes

        if not temp_key == key:
          temp_key = key
          temp_flow_counter = 0
          flow_counter = 0
          if direction == '>':
            IsFlip = "no"
          else:
            IsFlip = "yes"
        else:
          temp_flow_counter = flow_counter
        
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
        if direction == '>':
          total_to_packets += packet
          total_to_bytes += byte
        else:
          total_from_packets += packet
          total_from_bytes += byte
        ts_last = timestamp

    if key == temp_key:
      yield temp_key, temp_flow_counter, ts_begin, ts_last, total_to_packets, total_to_bytes, total_from_packets, total_from_bytes

  def sort_flow(data):
    startTime = -1
    nUdpFlow = 0
    aggr_total_to_packets = 0
    aggr_total_to_bytes = 0
    aggr_total_from_packets = 0
    aggr_total_from_bytes = 0

    for key in sorted(data, key=lambda x: x[2]):
      if (startTime == -1):
        startTime = float(key[2])
        nUdpFlow += 1
        aggr_total_to_packets += key[4]
        aggr_total_to_bytes += key[5]
        aggr_total_from_packets += key[6]
        aggr_total_from_bytes += key[7]

      yield key[0], key[1],float(key[2])-startTime, float(key[3])-startTime, key[4], key[5], key[6] ,key[7], nUdpFlow, aggr_total_to_packets, aggr_total_to_bytes, aggr_total_from_packets, aggr_total_from_bytes

  def output(data):
    for key in data:
      k = '{} {}'.format(key[0],key[1])
      v = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(key[2], key[3], float(key[3])-float(key[2]), key[4], key[5], key[6] ,key[7], key[8], key[9], key[10], key[11], key[12])
      yield '{}\t{}\n'.format(k, v)
  
  return output(sort_flow(sum(load(input))))

if __name__ == '__main__':
  taskname = sys.argv[1]
  if taskname == 'mapper':
    sys.stdout.writelines(mapper(sys.stdin))
  elif taskname == 'reducer':
    sys.stdout.writelines(reducer(sys.stdin))
