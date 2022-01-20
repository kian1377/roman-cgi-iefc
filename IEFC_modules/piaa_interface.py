import numpy as np
from matplotlib import pyplot as plt
from datetime import date
import astropy.io.fits as pyfits
import time
from datetime import date

class PIAATestbedInterface():

    def __init__(self, testbed, dm_interface, lyot_server, voltage_flat, gain_path):
        self.testbed = testbed
        self.dmi = dm_interface
        
        ### log settings
        self.date = date.today().strftime("%Y%m%d")    
            
        ### DM Settings
        self.dm_reference = voltage_flat.copy() # use vflat as reference
        self.voltage_return = np.zeros(voltage_flat.shape) # dummy variable for dmcoef.set_dm
        self.dm_state = np.zeros(voltage_flat.shape)
        self.dm_delay = 0.7 # s
        
        # DM gain in m/V
        gain_m_per_v = pyfits.getdata(gain_path).astype('double')
        self.dm_gain = 1.0 / gain_m_per_v
        
        ### Cam settings
        # how to use corners: full shape: (2560, 2160)
        # remember that image is rotated 90 to make rows=table y, cols=table x
        num_pixels = 216
        w_half = num_pixels//2 # image window is 2*w rows x 2*w cols
        c0 = int(lyot_server.xc[0]); r0 = 2560-int(lyot_server.yc[0])
        self.corners = [[c0-w_half, c0+w_half],[r0-w_half, r0+w_half]]
        self._texp = 5e-3

        self.run_number = 0
        self.mode = 'run'
        self.log_messages = []
    
    def log(self, log_message):
        time_stamp = time.asctime( time.localtime() )
        self.log_messages.append(time_stamp + ': ' + log_message)
    
    def set_mode(self, new_mode):
        self.mode = new_mode
        
    def update_run_number(self):
        self.run_number += 1
    
    def write_log_to_file(self, filename):
        timestamp = date.today()
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        stamp = '{:d}{:d}{:d}'.format(year, month, day)
        with open(filename + '_{:s}.txt'.format(stamp), 'w+') as file_writer:
            for message in self.log_messages:
                file_writer.write(message + '\n')

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
    