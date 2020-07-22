#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:12:21 2020

@author: chi
"""
import socket
import TrustCal as tc
import WebScrapper as ws
import pandas as pd

#Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Bind the socket to the port
server_address = ('128.2.144.116', 5000)
print('listening on {} port {}'.format(*server_address))
sock.bind(server_address)

#Listen for incoming connections
sock.listen(10)

#Load all vectors and models
cv_vector_title, cv_vector_text, tfidf_vector_title, tfidf_vector_text = tc.loadVectors()
svm,rf,lr = tc.loadModels()

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
                news = ws.processUrl(request)
                
                #parse information into a dataframe
                article = {'title': [news.title],'text': [news.text]}
                df = pd.DataFrame(article)
                
                #calculate percentage of real
                title = news.title
                percentage = tc.calculate(df,cv_vector_title, cv_vector_text, tfidf_vector_title, tfidf_vector_text, svm, rf, lr)
                
                #package info into JSON format
                # Status Codes:
                # O - OK
                # C - Connection timeout
                # I - Invalid URL
                # N - Invalid News URL
                # U - Unkown Error
                # reply format: 1 byte status code + 3 byte percentage + addional data
                reply = 'O' + str(percentage).rjust(3) + 'title: ' + title
                print('Server Reply: ', reply)
                data = reply.encode()
                connection.sendall(data)
            else:
                print('client done', client_address)
                break
    except Exception as e:
        print('Lost connection or Processing issue')
        #print(str(e))
    finally:
        # Clean up the connection
        connection.close()
sock.close()
