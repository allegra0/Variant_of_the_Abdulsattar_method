import os
import json
import csv
import cv2
from tkinter import filedialog
import numpy as np
from math import *
import PIL
from PIL import Image
import binascii

import random
import math
from random import shuffle
import sys
import gmpy2
from time import time
from Crypto.Util.number import getPrime

def contrast(image):
    I=np.asarray(image)
    contrast=np.zeros((512,512),np.uint8)
    for k in range (512):
        for l in range (512):
            contrast[k][l]=I[k][l]*1.2
    data=Image.fromarray(contrast)
    return data

bin_repr = lambda s, coding="iso-8859-1": ' '.join('{0:08b}'.format(c) for c in s.encode(coding))

def comp (tab):
    resultat=""
    taille=len(tab)
    for i in range (taille-1):
        com="0"
        if tab[i]>=tab[i+1]:
            com="1"
        resultat+=com
    return resultat

def process(adresse):
    img = Image.open(adresse)
    imgNG = img.convert('L')
    im2 = imgNG.resize((512,512))
    return im2

def algostates (im2):
  
    sequence={}
    
    eigmax=[]
    nbr=512//9
    block=np.zeros((9,9))#initialisation d'un bloc
    sblock=np.zeros((3,3))#initialisation d'un sous bloc
    I=np.asarray(im2)
    hash_seq=""
    
    for i in range(nbr):
        for j in range (nbr):
           
            if i<nbr:
                if j<nbr:
                    o=0
                    for k in range(9*i,9*i+9,1):
                        r=0
                        for l in range(9*j,9*j+9,1):
                            
                            
                            block[o][r]=I[k][l]
                            
                            r=r+1
                            
                        o+=1
                    eigmax=[]
                    for m in range (3):
                        for n in range(3):
                            w=0
                            for p in range(m*3,3+m*3,1):
                                c=0
                                for s in range(n*3,3+n*3,1):
                                    
                                    sblock[w][c]=block[p][s]
                                    
                                    c+=1
                                w+=1
                            eigmax.append(max(np.linalg.eigvals(sblock)))#calcul et ajout d'une eigenvalue max d'un sous bloc à la liste
                            
                    hash_seq=comp(eigmax)
                    if hash_seq !="00000000":
                        
                    
                        input_string=int(hash_seq, 2);
                        
                    #Obtain the total number of bytes
                        total_bytes= (input_string.bit_length()+7) // 8
                    #print(Total_bytes)
                        input_array = input_string.to_bytes(total_bytes, "big")
                    #Convert the bytes to an ASCII value and display it on the output screen
                        ascii_value=input_array.decode('iso-8859-1')
                        hash_code=ord(ascii_value)
                    else:
                        hash_code=0
                    if hash_code not in sequence.keys():
                        sequence[hash_code] = []
                    sequence[hash_code].append([(i,j),0])
                 
    print(len(sequence.keys()))
    som=0
    for key in sequence.keys():
        som+=len(sequence[key])
    print(som)
    with open("sequences.json",'w') as monfichier:
        json.dump(sequence,monfichier)

def algostate (im2 ):
    
    sequence={}
    eigmax=[]
    nbr=512//3
    block=np.zeros((9,9))#initialisation d'un bloc
    sblock=np.zeros((3,3))#initialisation d'un sous bloc
    I=np.asarray(im2)
    hash_seq=""
    
    for i in range(nbr):
        for j in range (nbr):
           
            if i<nbr-2:
                if j<nbr-2:
                    o=0
                    for k in range(int(9*i/3),int(9*i/3+9),1):
                        r=0
                        for l in range(int(9*j/3),int(9*j/3+9),1):
                            
                            
                            block[o][r]=I[k][l]
                            
                            r=r+1
                            
                        o+=1
                    eigmax=[]
                    for m in range (3):
                        for n in range(3):
                            w=0
                            for p in range(m*3,3+m*3,1):
                                c=0
                                for s in range(n*3,3+n*3,1):
                                    
                                    sblock[w][c]=block[p][s]
                                    
                                    c+=1
                                w+=1
                            eigmax.append(max(np.linalg.eigvals(sblock)))#calcul et ajout d'une eigenvalue max d'un sous bloc à la liste
                            
                    hash_seq=comp(eigmax)
                    if hash_seq !="00000000":
                        
                    
                        input_string=int(hash_seq, 2);
                        
                    #Obtain the total number of bytes
                        total_bytes= (input_string.bit_length()+7) // 8
                    #print(Total_bytes)
                        input_array = input_string.to_bytes(total_bytes, "big")
                    #Convert the bytes to an ASCII value and display it on the output screen
                        ascii_value=input_array.decode('iso-8859-1')
                        hash_code=ord(ascii_value)
                    else:
                        hash_code=0
                    if hash_code not in sequence.keys():
                        sequence[hash_code] = []
                    sequence[hash_code].append([(i,j),0])
                 
    print(len(sequence.keys()))
    #som=0
    #for key in sequence.keys():
     #   som+=len(sequence[key])
    
    with open("C:\\Users\\Administrateur\\\sequence.json",'w') as monfichier:
    
    #with open("C:\\Users\\Administrateur\\state_art1\\sequence\\sequence"+str(rang)+".json",'w') as monfichier:
        json.dump(sequence,monfichier)
        
def stateart (im2 ):# c'est la meme chose que algostate() juste qu'ici les blocs sont comptés selon un ordre raster alors que dans algostate c'est selon les coordonnées du premier pixel gauche en haut
    
    sequence={}
    eigmax=[]
    nbr=512//3
    block=np.zeros((9,9))#initialisation d'un bloc
    sblock=np.zeros((3,3))#initialisation d'un sous bloc
    I=np.asarray(im2)
    hash_seq=""
    
    for i in range(nbr):
        for j in range (nbr):
           
            if i<nbr-2:
                if j<nbr-2:
                    o=0
                    for k in range(int(9*i/3),int(9*i/3+9),1):
                        r=0
                        for l in range(int(9*j/3),int(9*j/3+9),1):
                            
                            
                            block[o][r]=I[k][l]
                            
                            r=r+1
                            
                        o+=1
                    eigmax=[]
                    for m in range (3):
                        for n in range(3):
                            w=0
                            for p in range(m*3,3+m*3,1):
                                c=0
                                for s in range(n*3,3+n*3,1):
                                    
                                    sblock[w][c]=block[p][s]
                                    
                                    c+=1
                                w+=1
                            eigmax.append(max(np.linalg.eigvals(sblock)))#calcul et ajout d'une eigenvalue max d'un sous bloc à la liste
                            
                    hash_seq=comp(eigmax)
                    if hash_seq !="00000000":
                        
                    
                        input_string=int(hash_seq, 2);
                        
                    #Obtain the total number of bytes
                        total_bytes= (input_string.bit_length()+7) // 8
                    #print(Total_bytes)
                        input_array = input_string.to_bytes(total_bytes, "big")
                    #Convert the bytes to an ASCII value and display it on the output screen
                        ascii_value=input_array.decode('iso-8859-1')
                        hash_code=ord(ascii_value)
                    else:
                        hash_code=0
                    if hash_code not in sequence.keys():
                        sequence[hash_code] = []
                    sequence[hash_code].append([167*i+j,0])
                 
    #print(len(sequence.keys()))
    #som=0
    #for key in sequence.keys():
     #   som+=len(sequence[key])
    
    with open("C:\\Users\\Administrateur\\state_art1\\sequencestate.json",'w') as monfichier:
    
    #with open("C:\\Users\\Administrateur\\state_art1\\sequence\\sequence"+str(rang)+".json",'w') as monfichier:
        json.dump(sequence,monfichier)


def gcd(a,b):
    while b > 0:
        a, b = b, a % b
    return a
    
def lcm(a, b):
    return a * b // gcd(a, b)    
    
    
def int_time():
    return int(round(time() * 1000))

class PrivateKey(object):
    def __init__(self, p, q, n):
        #self.l = lcm(p-1,q-1)----This is added as requested by the setup BUT not used, shortcut is used!
        self.l = (p-1) * (q-1)
        #self.m = gmpy2.invert(gmpy2.f_div(gmpy2.sub(gmpy2.powmod(n+1,self.l,n*n),gmpy2.mpz(1)),pub.n),n) --- Shortcut used instead of it
        self.m = gmpy2.invert(self.l, n)  #1/fi(n)
    def __repr__(self):
        #return '<PrivateKey: %s %s>' % (self.l, self.m)
        return self.l, self.m

class PublicKey(object):

    @classmethod
    def from_n(cls, n):
        return cls(n)
    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
    def __repr__(self):
        #return '<PublicKey: %s>' % self.n
        return self.n
    
def generate_keypair(bits):
    p_equal_q = True
    while p_equal_q:
        p = getPrime(bits // 2)
        q = getPrime(bits // 2)
        if (p!=q):
            p_equal_q = False
    n = p * q
    return (p-1, q-1), n

def encrypt(pub, plain):
   
    r = random.randrange(1,pub)
    while gcd(r,pub) != 1:
        r = random.randrange(1,pub)
    x = gmpy2.powmod(r,pub,pub*pub)
    cipher = gmpy2.f_mod(gmpy2.mul(gmpy2.powmod(pub+1,plain,pub*pub),x),pub*pub)
    
    return cipher

def decrypt(priv, pub, cipher):
    #r=gmpy2.powmod(cipher,(pub-1)%priv[0]*priv[-1],pub)
    #plain = gmpy2.mul(cipher,gmpy2.f_div(cipher,gmpy2.powmod(r,pub,pub))
    x = gmpy2.sub(gmpy2.powmod(cipher,priv[0]*priv[-1],pub*pub),1)
    plain = gmpy2.f_mod(gmpy2.mul(gmpy2.f_div(x,pub),gmpy2.invert(priv[0]*priv[-1], pub)),pub)
    return plain

def addemup(pub, a, b):
    return gmpy2.mul(a,b)

def multime(pub, a, n):
    return gmpy2.powmod(a, n, pub.n_sq)