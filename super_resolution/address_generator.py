import numpy as np
import pickle
import matplotlib.pyplot as plt

from scipy.misc import imresize

class AddressGenerator:
    
    def __init__(self, companies_path: str, addresses_path: str, logos_path: str):
        
        with open(companies_path, 'rb') as f:
            self.companies = pickle.load(f)
        
        with open(addresses_path, 'rb') as f:  
            self.addresses = pickle.load(f)
    
        with open(logos_path, 'rb') as f:  
            self.logos = pickle.load(f)
            
        self._styles = ['normal', 'italic', 'oblique']
        self._weight = ['normal', 'bold']
        self._font_family = ['Tahoma', 'DejaVu Sans', 'Arial', 'Verdana', 'sans-serif', 'serif',
                            'Times New Roman', 'Courier New', 'Georgia']
        
    def _sample(self, n: int) -> np.ndarray:
        
        # Draw companies - cut name off at 30 characters
        companies_draw = np.random.choice(self.companies, n)
        for i in range(len(companies_draw)):
            companies_draw[i] = companies_draw[i][:35]
            
        addresses_draw = np.random.choice(self.addresses, n)
        
        logo_rand_idx = np.random.randint(self.logos.shape[0], size = (n, ))
        logos_draw = self.logos[logo_rand_idx]
        
        for e in range(n):
            
            if 'road' not in addresses_draw[e]:
                addresses_draw[e]['road'] = 'Placeholder Road 10'
                
            if 'zip' not in addresses_draw[e]:
                addresses_draw[e]['zip'] = '1000'
            
            addresses_draw[e]['address'] = companies_draw[e] + '\n' + \
                addresses_draw[e]['road'] + '\n' + \
                addresses_draw[e]['zip']
            addresses_draw[e]['logo'] = logos_draw[e]
                
            del addresses_draw[e]['zip'], addresses_draw[e]['road']
                
        return addresses_draw
    
    def generate_header(self, n: int):
        
        # Draw element
        elements = self._sample(n)
        
        headers_hr = []
        headers_lr = []
        for element in elements:
            
            # Make figure
            dpi = 150
            # We add to the size such that we hit the right dimensions when cropping later
            fig = plt.figure(figsize = (224 * 2 / dpi, 224 * 1.5 / dpi), dpi = dpi)
            # Subplot
            ax = fig.add_subplot(111)
            
            # Define axis
            ax.axis([0, 60, 0, 20])
            
            # Add text
            ax.text(
                58, 8, element['address'],
                fontsize = 6,
                style = np.random.choice(self._styles),
                weight = np.random.choice(self._weight),
                family = np.random.choice(self._font_family),
                horizontalalignment = 'right'#,
                #bbox = {'facecolor':'white', 'pad':8, 'linewidth': 0.5}
            )
            
            # Add logo
            ax.imshow(element['logo'], aspect ='equal', extent = (2, 10, 7, 15))
            
            # Add horizontal line
            ax.axhline(3, color = 'black', linewidth = 0.5)
            
            # Hide axis
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Hide border
            fig.gca().set_frame_on(False)
            
            # Draw picture. Unfortunately, matplotlib seems to produce low quality
            # figure when just rendering the image. As a hack we have to save it to
            # disk
            #fig.canvas.draw()
            fig.savefig('tmp.png', dpi = dpi)
            
            # Extract as numpy pixel array
            hr_numpy = np.array(fig.canvas.renderer._renderer)[:,:,:3]
            
            # Crop image to 88x288 dimension
            hr_numpy = hr_numpy[56:336 - 56, 52:448 - 40, :]
            
            # Resize to low resolution
            resize_factor = tuple([int(num / 2) for num in hr_numpy.shape[:2]])
            lr_numpy = imresize(hr_numpy, size = resize_factor)
            
            headers_hr.append(hr_numpy)
            headers_lr.append(lr_numpy)
            plt.close()
            
        self.headers_hr = np.stack(headers_hr, axis = 0)
        self.headers_lr = np.stack(headers_lr, axis = 0)
        