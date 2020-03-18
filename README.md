# Collective Mode Brownian Dynamics
A HOOMD-blue plugin for Collective Mode Brownian Dynamics to accelerate the equilibrium sampling of soft matter systems. CMBD has been tested to work with HOOMD 2.4 in single precision on Linux.

See the following paper for details about the theory and implementation:

K. S. Silmore and J. W. Swan, ["Collective mode Brownian dynamics: A method for fast relaxation of statistical ensembles"](https://doi.org/10.1063/1.5129648), J. Chem. Phys. 152, 094104 (2020). [doi: 10.1063/1.5129648](https://doi.org/10.1063/1.5129648)


## HOOMD-blue Installation

HOOMD-blue installation instructions and software requirements can can be found [here](https://hoomd-blue.readthedocs.io/en/v2.5.0/compiling.html). After creating the build directory and running `cmake`, you should run the command `ccmake .` inside the build directory and ensure that the following options are enabled:
```
COPY_HEADERS            ON
ENABLE_CUDA             ON
SINGLE_PRECISION        ON
```
Specifically, the `COPY_HEADERS` option is necessary to compile the plugin.

Depending on where you intalled HOOMD, you may need to point python to its location by updating the `PYTHONPATH` environment variable in your terminal session or your `.bashrc` file:
```
export PYTHONPATH=$PYTHONPATH:[HOOMD_INSTALL]
```

## Plugin Installation

After downloading the plugin code, edit the following line in the `FindHOOMD.cmake` so that HOOMD can be found and relevant compilation configuration settings can be automatically detected:
```
set(HOOMD_ROOT [HOOMD_INSTALL]/hoomd CACHE FILEPATH "Directory containing a hoomd installation (i.e. _hoomd.so)")
```
Alternatively, you can issue the `cmake` command below with the option `-DHOOMD_ROOT=[HOOMD_INSTALL]/hoomd`.

Next, issue the following commands to create a build directory, compile the plugin, and install the plugin in the hoomd installation directory:
```
mkdir build
cd build
cmake ../
make -j4
make install
```

## Example Python Code
CMBD can be called like most of the other integration schemes in HOOMD. 
```python
import hoomd.md
import numpy as np
import hoomd.collective_mode

...

# CMBD strength
alpha = 0.5

# define a set of 3 wavevectors to excite of wavenumber kval
ks = np.array([[0,0,kval],[0,kval,0],[kval,0,0]])

hoomd.md.integrate.mode_standard(dt=dt)
hoomd.collective_mode.integrate.collective(group=group.all(), kT=kT, ks=ks, seed=seed, alpha=alpha)

```

## Current Limitations
* Requires CUDA (no CPU implementation)
* Simulation box must be rectangular and must not change throughout the course of the simulation

