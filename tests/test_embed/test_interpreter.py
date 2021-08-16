# -*- coding: utf-8 -*-
import sys

from widget_module import Widget


class DerivedWidget(Widget):
    def __init__(self, message):
        super(DerivedWidget, self).__init__(message)

    def the_answer(self):
        return 42

    def argv0(self):
        return sys.argv[0]
