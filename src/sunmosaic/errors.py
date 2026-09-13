"""Exception types with actionable, user-facing messages."""


class SunMosaicError(Exception):
    """Base class for all errors raised by SunMosaic."""


class InputError(SunMosaicError):
    """The supplied files cannot be used (count, format, size)."""


class RegistrationError(SunMosaicError):
    """The tiles could not be placed relative to each other."""
