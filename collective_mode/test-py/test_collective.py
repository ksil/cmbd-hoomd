# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

import hoomd;
hoomd.context.initialize();
import hoomd.collective_mode;
import unittest;
import os;

class test_simple(unittest.TestCase):
    def test_constructor(self):
        sysdef = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[1,2]);

        updater = hoomd.collective_mode.update.example(4);

if __name__ == '__main__':
    unittest.main(argv = ['test_example.py', '-v'])
