#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/8
# @Author  : Wenhao Shan


class ActionError(BaseException):
    """
    raise exception with message input
    """
    message = None

    def __init__(self, message):
        self.message = message

