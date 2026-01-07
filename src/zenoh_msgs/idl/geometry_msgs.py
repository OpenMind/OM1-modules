from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import array, float32, float64

from .std_msgs import Header


@dataclass
class Point(IdlStruct, typename="Point"):
    """
    Representation of a point in 3D space with float64 precision.
    """

    x: float64
    y: float64
    z: float64


@dataclass
class Point32(IdlStruct, typename="Point32"):
    """
    Representation of a point in 3D space with float32 precision.
    """

    x: float32
    y: float32
    z: float32


@dataclass
class Quaternion(IdlStruct, typename="Quaternion"):
    """
    Representation of an orientation in free space.
    """

    x: float64
    y: float64
    z: float64
    w: float64


@dataclass
class Pose(IdlStruct, typename="Pose"):
    """
    Representation of a pose in free space.
    """

    position: Point
    orientation: Quaternion


@dataclass
class PoseStamped(IdlStruct, typename="PoseStamped"):
    """
    Stamped pose data.
    """

    header: Header
    pose: Pose


@dataclass
class PoseWithCovariance(IdlStruct, typename="PoseWithCovariance"):
    """
    Pose with covariance.
    """

    pose: Pose
    covariance: array[float64, 36]


@dataclass
class PoseWithCovarianceStamped(IdlStruct, typename="PoseWithCovarianceStamped"):
    """
    Stamped pose with covariance.
    """

    header: Header
    pose: PoseWithCovariance


@dataclass
class Vector3(IdlStruct, typename="Vector3"):
    """
    Representation of a vector in free space.
    """

    x: float64
    y: float64
    z: float64


@dataclass
class Twist(IdlStruct, typename="Twist"):
    """
    Twist represents velocity in free space broken into its linear and angular parts.
    """

    linear: Vector3
    angular: Vector3


@dataclass
class TwistWithCovariance(IdlStruct, typename="TwistWithCovariance"):
    """
    Twist with covariance.
    """

    twist: Twist
    covariance: array[float64, 36]


@dataclass
class TwistWithCovarianceStamped(IdlStruct, typename="TwistWithCovarianceStamped"):
    """
    Stamped twist with covariance.
    """

    header: Header
    twist: TwistWithCovariance


@dataclass
class Accel(IdlStruct, typename="Accel"):
    """
    Acceleration in free space.
    """

    linear: Vector3
    angular: Vector3


@dataclass
class AccelWithCovariance(IdlStruct, typename="AccelWithCovariance"):
    """
    Acceleration with covariance.
    """

    accel: Accel
    covariance: array[float64, 36]


@dataclass
class AccelWithCovarianceStamped(IdlStruct, typename="AccelWithCovarianceStamped"):
    """
    Stamped acceleration with covariance.
    """

    header: Header
    accel: AccelWithCovariance
