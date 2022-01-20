import numpy as np
from matplotlib import pyplot as plt
from datetime import date
import astropy.io.fits as pyfits
import time
from datetime import date

from poppy_roman_cgi_phasec import hlc
import misc

class CGI():

    def __init__(self, cgi_mode='hlc'):
        
        self.cgi_mode = cgi_mode
        
        self.cgi_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data')
        self.dm_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data/dms')
        
        self.Nact = 48
        self.dm_diam = 46.3*u.mm
        self.act_spacing = 0.9906*u.mm
        
        self.DM1 = poppy.ContinuousDeformableMirror(name='DM1', dm_shape=(Nact,Nact), actuator_spacing=act_spacing, 
                                               radius=dm_diam/2, influence_func=str(dm_dir/'proper_inf_func.fits'))
        self.DM2 = poppy.ContinuousDeformableMirror(name='DM2', dm_shape=(Nact,Nact), actuator_spacing=act_spacing, 
                                               radius=dm_diam/2, influence_func=str(dm_dir/'proper_inf_func.fits'))
        
        
        
    
    def log(self, log_message):
        time_stamp = time.asctime( time.localtime() )
        self.log_messages.append(time_stamp + ': ' + log_message)
    
    def set_mode(self, new_mode):
        self.mode = new_mode

    def set_shutter(self, shutter_state):
        if shutter_state:
            self.testbed.nkt_onoff(1)
            self.testbed.nkt_onoff(1)
        else:
            self.testbed.nkt_onoff(0)
            self.testbed.nkt_onoff(0)
    
    @property
    def delay(self):
        return self.dm_delay
    
    @delay.setter
    def delay(self, delay):
        self.dm_delay = delay
        
    @property
    def texp(self):
        return self._texp
    
    @texp.setter
    def texp(self, texp):
        self._texp = texp
    
    def reset_dm(self):
        self.dm_state = np.zeros(self.dm_reference.shape)

    def set_dm(self, dm_command):
        self.dm_state = self.dm_gain * dm_command.reshape((34,34))
    
    def add_dm(self, dm_command):
        self.dm_state += self.dm_gain * dm_command.reshape((34,34))
               
    def get_dm_state(self):
        return self.dm_state
    
    def send_dm(self):    
        ret = self.dmi.set_dm(self.dm_reference + self.dm_state, self.voltage_return, delay=self.dm_delay)
        return ret
        
    def snap(self):
        subdir = 'iefc_{:s}_run_{:d}_{:s}'.format(self.date, self.run_number, self.mode)
            
        ret = self.testbed.snap(self.texp, subdir=subdir, param={'corners':self.corners})
        return ret['img'].ravel()
        