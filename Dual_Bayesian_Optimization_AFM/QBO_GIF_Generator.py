import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from QBO_Image_Generator import BayesianImageGenerator
        
def BayesianGIFGenerator(file_path,model,mode,n_start,n_end,n_frames,gif_filename,gif_duration):
    """
    mode: 'Value' or 'Uncertainty'
    model: 'Quantum' or 'Classical'
    """
    TEMP_DIR = 'GIFGenerator_Folder'
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    image_filenames = []
            
    model = BayesianImageGenerator(
        file_path=file_path,
        model=model,
        mode=mode,
        n_start=n_start,
        n_end=n_end
        )
    
    model.run_optimizer()
        
    step = max(1, n_end // n_frames)
    
    for i in np.arange(n_start,n_end+1,step):
        filename = os.path.join(TEMP_DIR, f'iteration_{i:03d}.png')
        fig = model.plot_iteration(i)
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close(fig)
        image_filenames.append(filename)
        print(f'Frame {i:03d} saved')
        
    frame_duration = gif_duration / len(image_filenames)
    with imageio.get_writer(gif_filename, mode='I', duration=frame_duration) as writer:
        for filename in image_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in image_filenames:
        os.remove(filename)
    
    print("Gif made succesfully")
    
    