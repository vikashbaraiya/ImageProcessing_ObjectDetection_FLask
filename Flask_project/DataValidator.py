import re

from flask import request


class DataValidator:

    @classmethod 
    def isfileEx(self, val):
        if re.match("^(([a-zA-Z]:)|(\\{2}\w+)\$?)(\\(\w[\w].*))(.jpg|.JPG|.png|.PNG|.jpeg|.JPEG|)$", val):
            return False
        else:
            return True


    def input_validation(self):
        inputError =  self.request.form["name"]
        if(DataValidator.isfileEx(self.form["firstName"])):
            inputError["firstName"] = "First Name can not be null"
            self.request.form["error"] = True
        return 