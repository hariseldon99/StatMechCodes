#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:01:27 2020

@author: daneel
"""
# Changes by A. Roy @ https://github.com/StatMechCodes/Metrop_Ising.ipynb

import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt
from tqdm import tqdm

#Credited to Alex McFarlane @ https://flipdazed.github.io/
class Periodic_Lattice(np.ndarray):
    """Creates an n-dimensional ring that joins on boundaries w/ numpy
    
    Required Inputs
        array :: np.array :: n-dim numpy array to use wrap with
    
    Only currently supports single point selections wrapped around the boundary
    """
    def __new__(cls, input_array, lattice_spacing=None):
        """__new__ is called by numpy when and explicit constructor is used:
        obj = MySubClass(params) otherwise we must rely on __array_finalize
         """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        
        # add the new attribute to the created instance
        obj.lattice_shape = input_array.shape
        obj.lattice_dim = len(input_array.shape)
        obj.lattice_spacing = lattice_spacing
        
        # Finally, we must return the newly created object:
        return obj
    
    def __getitem__(self, index):
        index = self.latticeWrapIdx(index)
        return super(Periodic_Lattice, self).__getitem__(index)
    
    def __setitem__(self, index, item):
        index = self.latticeWrapIdx(index)
        return super(Periodic_Lattice, self).__setitem__(index, item)
    
    def __array_finalize__(self, obj):
        """ ndarray.__new__ passes __array_finalize__ the new object, 
        of our own class (self) as well as the object from which the view has
        been taken (obj). 
        """
        # ``self`` is a new object resulting from
        # ndarray.__new__(Periodic_Lattice, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. Periodic_Lattice():
        #   1. obj is None
        #       (we're in the middle of the Periodic_Lattice.__new__
        #       constructor, and self.info will be set when we return to
        #       Periodic_Lattice.__new__)
        if obj is None: return
        #   2. From view casting - e.g arr.view(Periodic_Lattice):
        #       obj is arr
        #       (type(obj) can be Periodic_Lattice)
        #   3. From new-from-template - e.g lattice[:3]
        #       type(obj) is Periodic_Lattice
        # 
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'spacing', because this
        # method sees all creation of default objects - with the
        # Periodic_Lattice.__new__ constructor, but also with
        # arr.view(Periodic_Lattice).
        #
        # These are in effect the default values from these operations
        self.lattice_shape = getattr(obj, 'lattice_shape', obj.shape)
        self.lattice_dim = getattr(obj, 'lattice_dim', len(obj.shape))
        self.lattice_spacing = getattr(obj, 'lattice_spacing', None)
        pass
    
    def latticeWrapIdx(self, index):
        """returns periodic lattice index 
        for a given iterable index
        
        Required Inputs:
            index :: iterable :: one integer for each axis
        
        This is NOT compatible with slicing
        """
        if not hasattr(index, '__iter__'): return index         # handle integer slices
        if len(index) != len(self.lattice_shape): return index  # must reference a scalar
        if any(type(i) == slice for i in index): return index   # slices not supported
        if len(index) == len(self.lattice_shape):               # periodic indexing of scalars
            mod_index = tuple(( (i%s + s)%s for i,s in zip(index, self.lattice_shape)))
            return mod_index
        raise ValueError('Unexpected index: {}'.format(index))

        
# Original code by B.D. Hammel @ https://github.com/bdhammel/ising-model
class IsingLattice:

    def __init__(self, temperature=1.0,\
                 field=0.0,\
                 max_epochs=10,\
                 initial_state=np.random.choice([-1, 1], size = (4,4))):
        """Build the system from an initial state
        Build either a randomly distributed system or a homogeneous system (for
        watching the deterioration of magnetization
        Parameters
        ----------
        temperature   : Provided in units where k_B = J = 1. Default = 1
        field         : External field in units of J. Default = 0.
        max_epochs    : The maximum number of monte carlo steps. Default = 100
        initial_state : numpy array with two axes. Any size or shape will do.
                        For 1d problems, choose a shape of (1,N) or (N,1). 
                        Default = random 4X4
        """
        if initial_state.ndim != 2:
            raise ValueError("Currently only 2d arrays (of any shape) are supported.\
                            For For 1d problems, choose a shape of (1,N) or (N,1)")
        self.shape = initial_state.shape
        self.rows, self.cols = self.shape
        self.graph = np.meshgrid(np.arange(self.rows), np.arange(self.cols))
        self.size = np.prod(self.shape)
        self.T = temperature
        self.h = field
        self.max_epochs = max_epochs
        self.system = Periodic_Lattice(initial_state)
        #Initialize all thermodynamic data as functions of epoch (time) with blank lists
        self.epochdata = {"epochs":[], "mags":[], "chis":[], "energies":[], "cvs":[]}
            

    def local_energy_spins(self, N, M):
        """Calculate the energy of spin interaction at a given lattice site
        i.e. the interaction of a Spin at lattice site n,m with its 4 neighbors
        - S_n,m * (S_n+1,m + Sn-1,m + S_n,m-1, + S_n,m+1 )
        Note that, for a 1d lattice, the interaction changes to
        - S_1,m * (S_1,m+1 + S_1, m-1 ), or
        - S_m,1 * (S_m+1, 1 + S_m-1,1)
        Parameters
        ----------
        N : int
            lattice site coordinate
        M : int
            lattice site coordinate
        Return
        ------
        float
            energy of the site
        """
        rows, cols = self.shape
        if rows == 1:
            e = - self.system[N,M] * (self.system[N, M+1] + self.system[N, M-1] )
        elif cols == 1:
            e = - self.system[N,M] * (self.system[N+1, M] + self.system[N-1, M])
        else:
            e = - self.system[N, M]*(
            self.system[N - 1, M] + self.system[N + 1, M]
            + self.system[N, M - 1] + self.system[N, M + 1]
        )
        return e
    
    def local_energy_field(self, N, M):
        """Calculate the energy of field interaction at a given lattice site
        i.e. -S_nm * h
        Parameters
        ----------
        N : int
            lattice site coordinate
        M : int
            lattice site coordinate
        Return
        ------
        float
            energy of the site
        """
        return - self.system[N,M] * self.h

    def energy(self, N, M):
        """Calculate the energy at a given lattice site
        Parameters
        ----------
        N : int
            lattice site coordinate
        M : int
            lattice site coordinate
        Return
        ------
        float
            energy of the site
        """
        return self.local_energy_spins(N,M) + self.local_energy_field(N,M)
        
    @property
    def internal_energy(self):
        i, j = self.graph
        #The 0.5 is to offset for double counting the bonds during summation
        energies = 0.5 * self.local_energy_spins(i,j) +\
                                                self.local_energy_field(i,j)
        U = np.sum(energies.flatten())/self.size
        U_2 =  np.sum(energies.flatten()**2)/self.size
        return U, U_2

    @property
    def magnetization(self):
        """Find the average magnetization of the system
           Find the average mag^2 of the system also
        """
        m = np.sum(self.system)/self.size
        m_2 = np.sum(self.system**2)/self.size
        return m, m_2

    def metrop_gen(self):
        """Yield a generator for the metropolis simulation
        """
        for epoch in tqdm(np.arange(self.max_epochs)):
            # Randomly select a site on the lattice
            N, M = [np.random.randint(0, high=ndim) for ndim in self.shape]

            # Calculate energy of a flipped spin
            E = -1*self.energy(N, M)

            # "Roll the dice" to see if the spin is flipped
            if E <= 0.:
                self.system[N, M] *= -1
            elif np.exp(-E/self.T) > np.random.rand():
                self.system[N, M] *= -1
            #Calculate thermodynamic quantities
            m, msq = self.magnetization
            e, esq = self.internal_energy
            t = self.T
            n = self.size
            tsq = t * t
            nsq = n * n
            x = (msq - m*m)/(t * n)
            cv = (esq-e*e)/(tsq * nsq)

            #Update epochdata dict
            self.epochdata["epochs"].append(epoch)
            self.epochdata["mags"].append(m)
            self.epochdata["chis"].append(x)
            self.epochdata["energies"].append(e)
            self.epochdata["cvs"].append(cv)
            yield self 


def ising_video(lattice, plotter):
    """Prepare one frame of video for ising model simulation
    """
    #Get thermodynamics from continuously updated epochdata dict
    t = lattice.epochdata["epochs"]
    mags = lattice.epochdata["mags"]
    chis = lattice.epochdata["chis"]
    energies = lattice.epochdata["energies"]
    cvs = lattice.epochdata["cvs"]
    
    #Clear the plot of the previous frame and replot
    plotter.clf()
    # Get the figure and gridspec
    fig3 = plotter.gcf()
    grid = fig3.add_gridspec(4, 2)

    #Plot lattice on this axis
    lattice_ax = fig3.add_subplot(grid[0:,0])
    lattice_ax.set_title("T=%2.2lf, h=%2.2lf, size = %d" % (lattice.T,\
                                                           lattice.h,\
                                                           lattice.size))
    lattice_ax.axes.get_xaxis().set_visible(False)
    lattice_ax.axes.get_yaxis().set_visible(False)
    lattice_ax.imshow(lattice.system)

    #Plot thermodynamic quantities on these axes
    #Note that we're plotting time averages uptp the instant 
    mags_ax = fig3.add_subplot(grid[0,1])
    if  mags:
        mags_ax.set_title("m = %1.4lf" % np.average(mags))
        mags_ax.axes.get_xaxis().set_visible(False)
        mags_ax.plot(t, np.cumsum(mags)/np.arange(1,1+len(mags)))
            
    x_ax = fig3.add_subplot(grid[1,1])
    if  chis:    
        x_ax.set_title("X = %1.4lf" % np.average(chis))
        x_ax.axes.get_xaxis().set_visible(False)
        x_ax.plot(t, np.cumsum(chis)/np.arange(1,1+len(chis)))

    energy_ax = fig3.add_subplot(grid[2,1])    
    if  energies:
        energy_ax.set_title("e = %1.4lf" % np.average(energies))
        energy_ax.axes.get_xaxis().set_visible(False)
        energy_ax.plot(t, np.cumsum(energies)/np.arange(1,1+len(energies)))

    cv_ax = fig3.add_subplot(grid[3,1])    
    if  cvs:    
        cv_ax.set_title("Cv = %1.4lf" % np.average(cvs))
        cv_ax.set_xlabel("t (mc steps)")
        cv_ax.plot(t, np.cumsum(cvs)/np.arange(1,1+len(cvs)))


def ising_run(lattice, plotter,video=True, video_frate=60, **kwargs):
    """
    Runs the actual metropolis algorithm
    Parameters
        ----------
        Arguments:
        
        lattice     : IsingLattice object
        plotter     : matplotlib pyplot bject or figure
        
        Keywords:
        
        video       : Boolean for showing video of simulation, Default is True
        video_frate : Video frame rate in fps (frames per second), Default is 60
        
        Additional keyword arguments are passed to the matplotlib figure. 
        See their documentation for details.
    
        Return
        ------
        If video=False, then returns None
        Otherwise, it returns a matplotlib.animation object
        """
    if video:
        ani = anim.FuncAnimation(plotter.figure(**kwargs), ising_video, \
                                 frames=lattice.metrop_gen,\
                                 interval=1e3/video_frate,\
                                 fargs=(plotter,), save_count=lattice.max_epochs, repeat=False)
    else:
        ani = None
        #This runs the lattice metropolis generator
        for lattice in lattice.metrop_gen():
            pass
    return ani

if __name__ == '__main__':
    print("DONE!")

if __name__ == '__main__':

    fps = 15
    # Set up formatting for the movie files
    Writer = anim.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)
    
    #Size of fonts
    fs = 20
    plt.rcParams.update({'axes.titlesize': fs})
    plt.rcParams.update({'axes.labelsize':fs})
    
    #Size of figure
    plt.rcParams.update({'figure.figsize':(15,8)})
    plt.rcParams.update({'figure.autolayout':True})
    
    #Simulation Parameters
    lattice_shape = (10,10)
    maxtime = 5e4
    h = 0.01
    temp = 1.0
    
    #Initial Condition (random)
    ic = np.random.choice([-1, 1], size = lattice_shape)
    
    #Initiate the Ising system with the parameters above
    l = IsingLattice(initial_state=ic,\
                     temperature=temp,\
                     field=h, max_epochs=maxtime)
    
    
    #Run and display the Ising animation
    ani = ising_run(l, plt,video=True)
    
    #Uncomment this if you want to save the video to a file
    ani.save('ising_2d_video.mp4', writer=writer)
