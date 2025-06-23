TEMPLATE_REGISTRY = {}


def register_template(template_name):
    def decorator(cls):
        TEMPLATE_REGISTRY[template_name] = cls
        return cls

    return decorator
