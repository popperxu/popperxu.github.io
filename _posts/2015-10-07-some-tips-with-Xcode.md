---
layout: post
title: Xcode的一些小使用记录
categories: c++
tags: Xcode
---




## Xcode添加参数

c/c++里面
int main(int argc, char * argv[])
argc参数数目，argv[0]是文件名，argv[1]->argv[argc-1]是真正的参数

在Xcode里添加参数选项，只需要在菜单栏 product -> scheme -> edit scheme -> Run -> 填入参数



## Xcode working directory settings:

In Xcode go to Product > Scheme > Edit Scheme > Run test (on the right) > Options (middle top)

Down under Options check “Use custom working directory” and set it to the directory where your files are located.


