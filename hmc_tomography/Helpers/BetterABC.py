"""A small extension to the Abstract Base Class Meta object to allow abstract
attributes.

Created by StackOverflow user 'krassowksi' and distributed under the CC0 license.
Permalink: https://stackoverflow.com/a/50381071/6848887

Great thinking, dude.

To the extent possible under law, the person who associated CC0 with this work has
waived all copyright and related or neighboring rights to this work.

"""


from abc import ABCMeta as NativeABCMeta


class DummyAttribute:
    pass


def abstractattribute(obj=None):
    """Decorator to create abstract attributes."""
    if obj is None:
        obj = DummyAttribute()
    obj.__is_abstract_attribute__ = True
    return obj


class ABCMeta(NativeABCMeta):
    """Meta-class extending ABCMeta to allow for abstract attributes."""

    def __call__(cls, *args, **kwargs):
        instance = NativeABCMeta.__call__(cls, *args, **kwargs)

        abstract_attributes = set()
        for name in dir(instance):
            attribute_to_check = getattr(instance, name)
            try:
                is_abstract = getattr(
                    attribute_to_check, "__is_abstract_attribute__", False
                )
                if is_abstract:
                    abstract_attributes.add(name)
            except AttributeError:
                # This is to handle LASIF's custom getattr methods.
                pass

        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with"
                " abstract attributes: {}".format(
                    cls.__name__, ", ".join(abstract_attributes)
                )
            )
        return instance
