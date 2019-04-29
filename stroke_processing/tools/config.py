import os
import configparser

PWD = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(os.path.join(PWD, '..', '..', 'stroke.cfg'))

