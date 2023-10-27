from .base import *

ALLOWED_HOSTS = ["3.39.160.218", "www.aiphabtc.com", "aiphabtc.com"]

STATIC_ROOT = BASE_DIR / "static/"
STATICFILES_DIRS = []
DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'aiphabtc',
        'USER': 'dbmasteruser',
        'PASSWORD': '9^Mmi9x&gdaLop5C8|X`0ZUjUxPq42cv',
        'HOST': 'ls-c734f681295dcc3ef154bf196883ce0d354c1c4f.cxmhq2eoxqys.ap-northeast-2.rds.amazonaws.com',
        'PORT': '5432',
    }
}
