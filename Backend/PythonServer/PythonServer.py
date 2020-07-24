#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:12:21 2020

@author: chi
"""
print('Starting server...')
import socket
import ServerUtil as su

#Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Bind the socket to the port
server_address = ('128.2.144.116', 5000)
print('\nlistening on {} port {}'.format(*server_address))
sock.bind(server_address)

#Listen for incoming connections
sock.listen(10)

cnt = 0
while True:
    # Wait for a connection
    print('\nwaiting for connection ', cnt)
    connection, client_address = sock.accept()
    cnt += 1
    try:
        print('connected from', client_address)

        while True:
            data = connection.recv(1024)
            if data:
                
                #decode to string
                request = data.decode('utf-8')
                print('Client Request: ', request)
                
                #scrape
                news = su.download(request)

                # Status Codes:
                # O - OK
                # C - Connection Timeout
                # I - Invalid URL
                # N - Not News URL
                # P - Processing Error
                # reply format: 1 byte status code + 3 byte percentage(optional) + addional data(optional)
                if not news:
                    reply = 'I'
                elif not news.title or not news.text:
                    reply = 'N'
                else:
                    title, percentage = su.process(news)
                    if not title or percentage is -1:
                        reply = 'P'
                    else:
                        reply = 'O' + str(percentage).rjust(3) + 'Title: ' + title

                print('Server Reply: ', reply)
                data = reply.encode()
                connection.sendall(data)
            else:
                print('client done', client_address)
                break
    except Exception as e:
        print('Lost connection or Processing issue')
        print(str(e))
    finally:
        # Clean up the connection
        connection.close()
sock.close()
