{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue255;
\red46\green155\blue191;}
{\*\expandedcolortbl;\csgray\c100000;\cssrgb\c100000\c100000\c100000;\cssrgb\c0\c0\c0;\csgray\c100000;
\cssrgb\c21176\c67059\c79608;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\b\fs44 \cf0 \cb2 \expnd0\expndtw0\kerning0
Data Structures
\b0\fs32 \cb4 \
\
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 As we have seen in the previous lessons, Python supports the following data structures: 
\b lists
\b0 , 
\b dictionaries
\b0 , 
\b tuples
\b0 , 
\b sets
\b0 .\cb4 \
\

\b \cb2 When to use a \cf5 dictionary\cf0 :
\b0 \cb4 \
\cb2 - When you need a logical association between a 
\b key:value
\b0  pair.\cb4 \
\cb2 - When you need fast lookup for your data, based on a custom key.\cb4 \
\cb2 - When your data is being constantly modified. Remember, dictionaries are \cf5 mutable\cf0 .\cb4 \
\

\b \cb2 When to use the other types:
\b0 \cb4 \
\cb2 - Use 
\b lists 
\b0 if you have a collection of data that does not need random access. Try to choose lists when you need a simple, \cf5 iterable\cf0  collection that is modified frequently.\cb4 \
\cb2 - Use a 
\b set 
\b0 if you need uniqueness for the elements. \cb4 \
\cb2 - Use 
\b tuples 
\b0 when your data cannot change. 
\fs30 \cb2 \
\
\
Sets are like lists, with the difference that its elements can't be repeated. Tuples are also like lists, but they can't be modified. Dictionaries are basically lists made up of keys each associated with a value.}