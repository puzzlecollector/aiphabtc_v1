from django import template
from common.models import PointTokenTransaction

register = template.Library()

@register.simple_tag(takes_context=True)
def get_last_transactions(context):
    request = context['request']
    if not request.user.is_authenticated:
        return []
    # Make sure this query gets all types of transactions, not just tokens
    return PointTokenTransaction.objects.filter(
        user=request.user
    ).order_by('-timestamp')[:10]
