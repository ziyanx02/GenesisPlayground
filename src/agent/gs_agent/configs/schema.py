from pydantic import ConfigDict

# -----------------------------------------------------------------------------
# Pydantic
# -----------------------------------------------------------------------------


def genesis_pydantic_config(
    frozen: bool = False,
    arbitrary_types_allowed: bool = False,
) -> ConfigDict:
    """
    Default config for all Genesis Pydantic dataclasses and models
    https://docs.pydantic.dev/latest/api/config

    Usage:
        from pydantic.dataclasses import dataclass

        @dataclass(config=genesis_pydantic_config())
        class MyDataClass():
            ...
    """
    return ConfigDict(
        # forbid extra fields not defined in the model
        extra="forbid",
        # validate default values against field types
        validate_default=True,
        # don't protect any namespaces from field names
        protected_namespaces=(),
        # serialize enums using their values
        use_enum_values=True,
        # freeze the dataclass
        frozen=frozen,
        # allow arbitrary types (e.g., np.ndarray, torch.Tensor)
        arbitrary_types_allowed=arbitrary_types_allowed,
    )
