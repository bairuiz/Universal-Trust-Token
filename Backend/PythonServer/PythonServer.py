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
server_address = ('localhost', 5000)
print('listening on {} port {}'.format(*server_address))
sock.bind(server_address)

#Listen for incoming connections
sock.listen()

#Load all vectors and models
cv_vector_title, cv_vector_text, tfidf_vector_title, tfidf_vector_text = tc.loadVectors()
svm,rf,lr = tc.loadModels()

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('connection from', client_address)

        while True:
            data = connection.recv(1024)
            if data:
                
                #decode to string
                url = data.decode('utf-8')
                
                #scrape
                news = ws.processUrl(url)
                
                #parse information into a dataframe
                article = {'title': [news.title],'text': [news.text]}
                df = pd.DataFrame(article)
                
                #calculate percentage of real
                title = news.title
                percentage = tc.calculate(df,cv_vector_title, cv_vector_text, tfidf_vector_title, tfidf_vector_text, svm, rf, lr)
                
                #package info into JSON format
                reply = 'title: ' + title + ', percentage: ' + percentage
                data = reply.encode()
                connection.sendall(data)
            else:
                print('no data from', client_address)
                break
    except Exception as e:
        print(str(e))
        sock.close()
        connection.close()
        break
    finally:
        # Clean up the connection
        connection.close()
sock.close()