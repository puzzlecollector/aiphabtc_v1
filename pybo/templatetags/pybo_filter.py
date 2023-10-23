from django import template
import markdown
from django.utils.safestring import mark_safe
from datetime import datetime, timedelta


register = template.Library()

@register.filter
def sub(value, arg):
    return value - arg

@register.filter
def mark(value):
    extensions = ["nl2br", "fenced_code"]
    return mark_safe(markdown.markdown(value, extensions=extensions))




