from collections.abc import Iterable
import socket
import ctypes

def startSocket(ip, address):
  sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sckt.connect((ip, address))

  return sckt

def closeSocket(sckt):
  sckt.close()

def sendData(sckt, arr):

  sckt.send(arr)

def receiveData(sckt, num_int):

  total_byte_cnt = num_int * 4
  buf_size = 16 * 1024
  recv_byte_cnt = 0
  data = b''

  while recv_byte_cnt < total_byte_cnt:

    temp_data = sckt.recv(buf_size)
    data += temp_data
    recv_byte_cnt += len(temp_data)

  return data

def List2CArray(l):
  return (ctypes.c_int * len(l))(*l)

def CArray2List(b, num_int):
  lst = []
  total_byte_cnt = num_int * 4

  for i in range(0, total_byte_cnt, 4):
    lst.append(int.from_bytes(b[i:i+4], byteorder='little'))

  return lst

