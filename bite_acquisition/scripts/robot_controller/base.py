import abc

class RobotController(abc.ABC):
    @abc.abstractmethod
    def __init__(self, config):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def move_to_pose(self, pose):
        pass

    @abc.abstractmethod
    def move_to_acq_pose(self):
        pass

    @abc.abstractmethod
    def move_to_transfer_pose(self):
        pass