#!/usr/bin/env python3
import os
import glob
from tkinter import filedialog
from tkinter import *

def set_working_directory():
    master = Tk()
    master.directory =  filedialog.askdirectory()
    # Change working directory to the specified directory
    os.chdir( str( master.directory ) )
    print( 'Current working directory is: ',os.getcwd() )
    
def get_file_list( ):
    filenames = glob.glob( '*.nxspe' )
    print ('Number of files: ',len(filenames))
    return filenames


