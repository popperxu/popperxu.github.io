---
layout: post
title: Mac上python读写csv的分隔符问题
categories: python
tags: python
---

在mac上用python将数据写入csv文件，用excel打开总是全部写在同一列上。后来在stackoverflow上找到一个回复：

>
On Mac OS X, this setting seems to be deduced from the decimal separator setting (in the language pane of system preferences). If the decimal separator is a point then the default CSV separator will be a comma, but if the decimal separator is a comma, then the default CSV separator will be a semicolon.


也就是说这是因为系统偏好设置里的decimal separator导致的问题。如果偏好设置里分隔符是句号，那么csv分隔符默认是逗号；如果偏好设置里分隔符是逗号，那么csv分隔符默认是分号。

csv文件一般都是以逗号作为分隔符，所以要不然修改偏好设置的分隔符为句号，要不然python读写csv的时候分隔符设置为分号。