class State:

    """
    Represents a state in the grid.

    Attributes:
    x (int): The x-coordinate of the state.
    y (int): The y-coordinate of the state.
    state_type (str): The type of the state, which can be 'wall', 'normal', or 'goal'.
    """

    def __init__(self, x=0, y=0, state_type='normal'):

        """
        Initializes the State object with coordinates and type.

        Args:
        x (int): The x-coordinate of the state. Default is 0.
        y (int): The y-coordinate of the state. Default is 0.
        state_type (str): The type of the state. Default is 'normal'.
        """
        self.x = x
        self.y = y
        if self.check_type(state_type): 
            self.state_type = state_type        
        else:
            raise ValueError("Type not supported")

    def check_type(self, state_type):

        accepted_types = ['wall', 'normal', 'goal']
        if state_type in accepted_types:
            return True
        else:
            return False

    @property
    def state_type(self):
        return self._state_type

    @state_type.setter
    def state_type(self, state_type):

        if self.check_type(state_type):
            self._state_type = state_type
        else:
            raise ValueError("Type not supported")