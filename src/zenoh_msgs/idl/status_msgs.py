from dataclasses import dataclass
from enum import Enum

from pycdr2 import IdlStruct
from pycdr2.types import int8

from .std_msgs import Header, String


@dataclass
class AudioStatus(IdlStruct, typename="AudioStatus"):
    """
    Status of the robot's audio system.
    """

    class STATUS_MIC(Enum):
        """
        Status of the microphone.
        """

        DISABLED = 0
        READY = 1
        ACTIVE = 2
        UNKNOWN = 3

    class STATUS_SPEAKER(Enum):
        """
        Status of the speaker.
        """

        DISABLED = 0
        READY = 1
        ACTIVE = 2
        UNKNOWN = 3

    header: Header
    status_mic: int8
    status_speaker: int8
    sentence_to_speak: String


@dataclass
class CameraStatus(IdlStruct, typename="CameraStatus"):
    """
    Status of the robot's camera system.
    """

    class STATUS(Enum):
        """
        Status of the camera.
        """

        DISABLED = 0
        ENABLED = 1

    header: Header
    status: int8


@dataclass
class MotionStatus(IdlStruct, typename="MotionStatus"):
    """
    Status of the robot's motion system.
    """

    class CONTROL(Enum):
        """
        Control mode of the robot.
        """

        DISABLED = 0
        AI = 1
        JOYSTICK = 2
        TELEOPS = 3

    class ATTITUDE(Enum):
        """
        Attitude of the robot's posture.
        """

        SITTING = 0
        STANDING = 1

    class STATE(Enum):
        """
        State of the robot's motion.
        """

        STILL = 0
        MOVING = 1

    header: Header
    control: int8
    attitude: int8
    state: int8
