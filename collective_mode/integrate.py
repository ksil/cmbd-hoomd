# First, we need to import the C++ module. It has the same name as this module (collective_mode) but with an underscore
# in front
from hoomd.collective_mode import _collective_mode

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from
# hoomd_script
import hoomd
from hoomd import _hoomd
import hoomd.md
from hoomd.integrate import _integration_method
import numpy as np

class collective(_integration_method):

    def __init__(self, group, kT, ks, seed = 1, alpha = 0.95):
        hoomd.util.print_status_line();

        # initialize base class
        _integration_method.__init__(self);

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT);

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group);


        # ensure that the ks numpy array is the right number of dimensions
        if not isinstance(ks, np.ndarray):
            hoomd.context.msg.error("ks must be a Nk x D numpy array, where Nk < 500 and D is the number of dimensions.")
            raise RuntimeError("Error creating CollectiveMode")

        D = hoomd.context.current.system_definition.getNDimensions()

        if ks.ndim != 2 or ks.shape[1] != D or ks.shape[0] > 500:
            hoomd.context.msg.error("ks must be a Nk x D numpy array, where Nk < 500 and D is the number of dimensions.")
            raise RuntimeError("Error creating CollectiveMode")

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            # my_class = _collective_mode.CollectiveMode;
            hoomd.context.msg.error("Collective Mode BD not currently implemented for CPU.")
            raise RuntimeError("Error creating CollectiveMode")
        else:
            my_class = _collective_mode.CollectiveModeGPU;

        self.cpp_method = my_class(hoomd.context.current.system_definition,
                                   group.cpp_group,
                                   kT.cpp_variant,
                                   seed,
                                   ks,
                                   alpha);

        self.cpp_method.validateGroup()

        # store metadata
        self.group = group
        self.kT = kT
        self.seed = seed
        self.ks = ks
        self.alpha = alpha
        self.metadata_fields = ['group', 'kT', 'seed', 'ks', 'alpha']

    def set_ks(self, ks):
        # ensure that the ks numpy array is the right number of dimensions
        if not isinstance(ks, np.ndarray):
            hoomd.context.msg.error("ks must be a Nk x D numpy array, where Nk < 500 and D is the number of dimensions.")
            raise RuntimeError("Error creating CollectiveMode")

        D = hoomd.context.current.system_definition.getNDimensions()

        if ks.ndim != 2 or ks.shape[1] != D or ks.shape[0] > 500:
            hoomd.context.msg.error("ks must be a Nk x D numpy array, where Nk < 500 and D is the number of dimensions.")
            raise RuntimeError("Error creating CollectiveMode")

        self.ks = ks
        self.cpp_method.set_ks(ks)

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.cpp_method.set_alpha(alpha)