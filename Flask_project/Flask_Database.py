from app import db, app
from flask import Flask, render_template, flash, request, redirect, url_for

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine

#create the object of Flask
from sqlalchemy.orm import sessionmaker


app.config['SECRET_KEY'] = 'hardsecretkey'


#SqlAlchemy Database Configuration With Mysql
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/flask'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False







# Models
class Profile(db.Model):
    __tablename__ = 'profile'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(20), unique=False, nullable=False)
    last_name = db.Column(db.String(20), unique=False, nullable=False)
    age = db.Column(db.Integer, nullable=False)

    #repr method represents how one object of this datatable
    # will look like
    def __repr__(self):
        return f"Name : {self.first_name}, Age: {self.age}"




