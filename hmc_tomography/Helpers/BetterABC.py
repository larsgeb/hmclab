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
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), "__is_abstract_attribute__", False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with"
                " abstract attributes: {}".format(
                    cls.__name__, ", ".join(abstract_attributes)
                )
            )
        return instance
