import OleFileIO_PL
from struct import unpack
import numpy as np

class CXDReader:
    def __init__(self, filename):
        self._ole = OleFileIO_PL.OleFileIO(filename)
        self._n = unpack('i',self._ole.openstream('File Info/Field Count').read())[0]
        self._closed = False
        
    def get_number_of_frames(self):
        return self._n
        
    def get_frame(self, i):
        if not i < self._n:
            return None
        else:
            prefix = 'Field Data/Field %d' % (i+1)
            bits = unpack('d',self._ole.openstream(prefix+'/Details/Image_Depth').read())[0]
            dtype = None
            if(bits == 16):
                dtype = np.uint16
            height = int(np.round(unpack('d',self._ole.openstream(prefix+'/Details/Image_Height').read())[0]))
            width = int(np.round(unpack('d',self._ole.openstream(prefix+'/Details/Image_Width').read())[0]))
            data = self._ole.openstream(prefix+'/i_Image1/Bitmap 1').read()
            img = np.frombuffer(data, dtype=dtype).reshape((height,width))
            return img

    def close(self):
        self._ole.close()
        self._closed = True

    def is_closed(self):
        return self._closed
        
