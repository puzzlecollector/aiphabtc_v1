from .base import *

ALLOWED_HOSTS = []


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'localtest',
        'USER': 'postgres',
        'PASSWORD': 'pwd',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}