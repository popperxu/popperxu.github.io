---
layout: post
title: 百度OCR企业版API的python调用小代码
categories: ocr
tags: ocr
---

上周使用[百度自然场景OCR服务](http://apistore.baidu.com/apiworks/servicedetail/969.html)进行了一些图片的测试，记录一下。要使用该API，需要先注册一个百度账号，用相应API key申请应用。

首先import会用到的packages。base64用于图像编解码，cv2是opencv包，用于显示图像。

	# -*- coding: utf-8 -*-
	import urllib, urllib2, json
	import base64
	import cv2
	import numpy as np
	
	#set encoding
	import sys, os
	
由于百度OCR API只能处理jpg文件，于是定义一个从某路径下读取所有jpg文件的函数：

	def my_listdir(directory):
	    """A specialized version of os.listdir() that ignores non-jpg files. """
	    file_list = os.listdir(directory)
	    #print file_list
	    return [x for x in file_list
	            if (x.endswith('.jpg') or x.endswith('.JPG') or x.endswith('.jpeg')]

然后定义一个baidu_ocr函数调用百度OCR API服务，输入为filename，即文件名，返回一个json string。	API可以指定的参数封装在data字典中，有

- fromdevice: iphone, android, pc
- clientip
- detecttype: Locate, LocateRecognition
- languagetype: CHN_ENG, ENG
- imagetype: 1, 经过base64编码的图像；2, 图像原文件
- image: 图片，小于300K的jpg文件

当imagetype=1时，读取文件后，使用base64.b64encode进行编码，再使用urlencode编码。在下面的代码中，将your api-key替换为自己的api-key，最终返回的content是一个json string。


	def baidu_ocr(filename):
	    #print sys.getdefaultencoding() 
	    reload(sys) 
	    sys.setdefaultencoding('utf8')
	
	    url = 'http://apis.baidu.com/idl_baidu/baiduocrpay/idlocrpaid'
	
	    data = {}
	    data['fromdevice'] = "iphone"
	    data['clientip'] = "10.10.10.0"
	    data['detecttype'] = "LocateRecognize"
	    data['languagetype'] = "CHN_ENG"
	    data['imagetype'] = "1"
	
	    # read image
	    file = open(filename, 'rb')
	    image =  file.read()
	    file.close()
	
	    data['image'] = base64.b64encode(image)
	
	    decoded_data = urllib.urlencode(data)
	    req = urllib2.Request(url, data = decoded_data)
	
	    req.add_header("Content-Type", "application/x-www-form-urlencoded")
	    req.add_header("apikey", "your api-kay")
	
	    resp = urllib2.urlopen(req)
	    content = resp.read()
	
	    return content

然后定义了一个text_reader的函数，根据API返回的json string在图像画出检测框，并打印出所有输出结果。我尝试想用opencvde cv2.putText() 在检测框边上标出识别结果，但是opencv的putText好像不支持中文，输出会是一串问号。。。暂时没找到解决方法。所以只是标出检测框及其编号。
	
	def text_reader(print_type=True):
	    img_path = 'web'
	    file_list = my_listdir(img_path)
	    img_len = len(file_list)
	    #print file_list
	    for i in xrange(img_len):
	        img_name = file_list[i]
	        
	        image = cv2.imread(os.path.join(img_path, file_list[i]))
	        cv2.imwrite(os.path.join(img_path, "tmp_"+img_name), image)
	
	        content = baidu_ocr(os.path.join(img_path, "tmp_"+img_name))
	
	        # print result
	        if(content and print_type):
	            dic = json.loads(content)
	            out_uni = json.dumps(dic, indent=2, ensure_ascii=False)
	            print out_uni
	
	        # imshow result
	        if(content):
	            dic = json.loads(content)
	            results = dic['retData']
	            box_num = 0
	            for result in results:
	                word = result['word']
	
	                width, top, height, left = result['rect'].values()
	                width, top, height, left = int(width), int(top), int(height), int(left)
	                #print type(width)
	                cv2.rectangle(image, (left, top), (left+width, top+height), (0, 255, 0), 2)
	                cv2.putText(image, unicode(box_num), (left, top+height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
	                box_num += 1
	            cv2.imshow("image", image)
	            cv2.waitKey(0)
	            cv2.destroyAllWindows()
	
	
	if __name__ == '__main__':
	    text_reader()

总结一下这其中遇到的编码问题。

- 在python内查看与修改当前编码方式：

```
	import sys
	sys.getdefaultencoding()  # 默认编码方式，mac下默认为ascii
	reload(sys)
	sys.setdefaultencoding(‘utf-8’) # 设置编码方式为utf-8
```	

- 判断是否是字符串(basestring, include str and unicode)，判断是否是str或unicode方法

```
	isinstance(a, str)
	isinstance(a, unicode)
	isinstance(a, basestring)
```

- json loads() 与 dumps()

json.dumps() 将python数据结构（比如dict）转化为string（默认情况）或unicode。

```
	json.dumps(obj, ensure_ascii, encoding…)
	- obj python数据结构：dic, list ...
	- ensure_ascii = True(default)
		dumps return str
	-ensure_ascii = False
		dumps return unicode
	-encoding
		在obj进行转化之前，所有obj中的str都会先转化为unicode，即str.decode(encoding)
	-默认情况：
		str -> unicode -> json(unicode) -> str
```
所以如果想要得到汉字的完整输出，可以设置ensure_ascii=False，得到unicode形式的输出，而不是转义后的str。

```
	output_unicode = json.dumps(dic, indent=2, ensure_ascii=False)
	print output_unicode
```

对于过程 str -> unicode -> json(unicode) -> str，第一过程str -> unicode的decode在encoding参数下控制，json(unicode) -> str 这里的encoding是默认控制的。
这个默认控制不是encode，而是直接把unicode转义为ascii编码，这个ascii编码的内容是unicode。

