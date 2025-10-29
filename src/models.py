from dataclasses import dataclass
from enum import Enum

class AgeGroup(str, Enum):
    YOUNG = "young"
    PRE_PRESBYOPIC = "pre-presbyopic"
    PRESBYOPIC = "presbyopic"

class Disease(str, Enum):
    MYOPE = "myope"
    HYPERMETROPE = "hypermetrope"
    ASTIGMATIC = "astigmatic"

class TearRate(str, Enum):
    NORMAL = "normal"
    REDUCED = "reduced"

class LensType(str, Enum):
    NONE = "none"
    HARD = "hard"
    SOFT = "soft"

@dataclass
class DatabaseConfig:
    db_name: str
    db_user: str
    db_pass: str
    db_host: str
    db_port: int
