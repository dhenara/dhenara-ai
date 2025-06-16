from datetime import datetime
from typing import Annotated

from pydantic import PlainSerializer

# custom datetime type for serialization to ISO format
ISODateTime = Annotated[
    datetime,
    PlainSerializer(lambda dt: dt.isoformat(), return_type=str),
]
